"""What one question costs, in dollars and seconds, under each way of answering it.

    uv run python -m eval.cost_compare                  # the full run, full1

Needs the full run's database (docker compose start). No model is called: like
eval/effort_check.py it rebuilds learn_gate's out-of-fold cascade from the saved answers, then
prices every option on the same items. Writes results/runs/cost_compare_<id>.json, which the
README and the paper show.

Added on 3 Nov, after the run. For each option, per question:

- **Muse dollars** from its exact tokens at the prices in .env, with no discount for prompt
  tokens Muse's API served from its own cache (their share is recorded for full context).
- **GPU dollars** for Llama: H100 seconds at Modal's listed rate. Drafts are timed. The cascade's
  confidence calls (4 resamples and a 1-token self-check, in parallel after the draft, on a
  cached prefix) were not timed: they are charged the draft's time again, which overstates
  them, as each resample writes fewer tokens than the draft. Requests are charged as if each
  had the GPU to itself; a GPU serving 32 at once costs less per question.
- **Memory build:** the wall time of the run's ingest on the H100 (16 calls at a time), split
  over its questions, for every option that reads the memory. A team that asks more questions
  of the same code pays it once.
- **Latency:** wall seconds as the harness saw them, draft first, then Muse if escalated.
- **Simpler alternatives** on the same items: Muse with the raw pages retrieved (no
  compression), the hand-written rule gate, and escalating at random as many questions as the
  cascade does (mean of 1,000 draws).
- **An off-the-shelf router:** RouteLLM's BERT router (eval/routellm_scores.py) decides from the
  question alone, before any draft: questions it sends up go straight to Muse with the team's
  memory, the rest to Llama. Its threshold is the most accurate within the same budgets as the
  cascade, chosen on these items (which favours it).
- **Team scale:** the replay's pages ingested per engineer, priced at the full run's measured
  build rate (H100 seconds per ingest call).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json

import numpy as np
import psycopg

from eval.common import RESULTS
from eval.learn_gate import (BENCHES, CONFIDENCE_FEATURES, HEADLINE, Ledger, Route,
                             cascade_choose, cascade_out_of_fold, load_rows)
from gateway.config import Settings

H100_USD_PER_S = 0.001097  # Modal's listed H100 price, $3.95 an hour (GPU only)
DRAWS, SEED = 1000, 0
OUTPUT_LIMIT = 262_144     # Muse's output limit: answers that reach it are runaways
LOCAL = Route.LOCAL_MEMORY.value


async def ledger_facts(url: str, run_id: str, bench: str) -> tuple[float, int, float]:
    """The memory build's wall time (first to last ingest call) and its number of calls, and the
    share of full-context Muse's prompt tokens that Muse's own API served from its cache (priced
    here at full rate)."""
    async with await psycopg.AsyncConnection.connect(url) as conn:
        cur = await conn.execute(
            "SELECT extract(epoch FROM max(created_at) - min(created_at)), count(*) FROM usage"
            " WHERE route = 'ingest' AND user_id = %s", (f"eval-{run_id}-{bench}-llama_only",))
        seconds, calls = await cur.fetchone()
        cur = await conn.execute(
            "SELECT sum(cached_tokens)::float / nullif(sum(prompt_tokens), 0) FROM usage"
            " WHERE route = 'muse_direct' AND user_id = %s", (f"eval-{run_id}-{bench}-muse_direct",))
        (cached,) = await cur.fetchone()
    return float(seconds or 0.0), int(calls), float(cached or 0.0)


def auc(score: list[float], positive: list[bool]) -> float:
    """P(a random positive scores above a random negative), ties counted half."""
    s, y = np.asarray(score, float), np.asarray(positive, bool)
    pos, neg = s[y], s[~y]
    return float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


def team(build_s: float, calls: int) -> dict:
    """Ingest per engineer in the latest real replay, shared and alone, priced per ingest call."""
    usd_per_call = build_s / calls * H100_USD_PER_S
    out = {"usd_per_ingest_call": usd_per_call}
    for order in ("related", "random"):
        for baseline, label in (("rule_gate", "shared"), ("no_sharing", "alone")):
            path = max(RESULTS.glob(f"replay_{baseline}_{order}_*.csv"))
            rows = [r for r in csv.DictReader(path.open()) if r["simulated"] == "False"]
            out[f"{order}_{label}"] = [
                {"team_size": int(r["team_size"]), "pages_per_engineer": float(r["ingest_pages_per_user"]),
                 "ingest_usd_per_engineer": float(r["ingest_pages_per_user"]) * usd_per_call,
                 "muse_tokens_per_engineer": float(r["muse_tokens_per_user"]),
                 "ingest_seconds_per_engineer": float(r["ingest_seconds_per_user"])}
                for r in sorted(rows, key=lambda r: int(r["team_size"]))]
    return out


def option(name: str, right, muse_usd, gpu_s, seconds, escalated, *, memory: bool,
           build_s: float, note: str = "") -> dict:
    right, muse_usd, gpu_s, seconds = map(np.asarray, (right, muse_usd, gpu_s, seconds))
    build = build_s / len(right) if memory else 0.0
    gpu_usd = (gpu_s.mean() + build) * H100_USD_PER_S
    total = muse_usd.mean() + gpu_usd
    return {"name": name, "note": note, "accuracy": float(right.mean()),
            "escalated": float(np.mean(escalated)),
            "muse_usd_per_1k": 1000 * float(muse_usd.mean()),
            "gpu_usd_per_1k": 1000 * float(gpu_usd),
            "memory_build_usd_per_1k": 1000 * build * H100_USD_PER_S,
            "usd_per_1k": 1000 * float(total),
            "usd_per_1k_correct": 1000 * float(total / right.mean()),
            "seconds_median": float(np.median(seconds)), "seconds_mean": float(seconds.mean()),
            "seconds_p90": float(np.percentile(seconds, 90))}


async def main(args) -> None:
    settings = Settings()
    price = Ledger(None, 0.0, settings.muse_input_price_per_m, settings.muse_output_price_per_m).cost
    routes = [LOCAL, Route.ESCALATE_COMPACT.value, Route.ESCALATE_RAW.value]
    url = args.database or f"{settings.database_url.rsplit('/', 1)[0]}/gateway_{args.id.replace('-', '_')}_full"
    run = argparse.Namespace(id=args.id, n=args.n, seed=0, database=url, no_confidence=False)
    rows, refs, _, confident = await load_rows(run, routes, price)
    assert confident, "needs eval/confidence.py's probes"
    cascade_out_of_fold(rows, list(CONFIDENCE_FEATURES))
    saved = json.loads((RESULTS / f"gate_{args.id}.json").read_text())
    rng = np.random.default_rng(SEED)
    path = RESULTS / f"routellm_bert_{args.id}.jsonl"
    scores = ({(s["bench"], s["item"]): s["strong_win_rate"] for s in map(json.loads, path.open())}
              if path.exists() else {})

    report = {"run_id": args.id, "h100_usd_per_s": H100_USD_PER_S,
              "muse_usd_per_m": [settings.muse_input_price_per_m, settings.muse_output_price_per_m],
              "benchmarks": {}}
    for bench in BENCHES:
        rs = [r for r in rows if r["bench"] == bench]
        build_s, calls, muse_cached = await ledger_facts(url, args.id, bench)
        if bench == "swebench":
            report["team"] = team(build_s, calls)
        draft_s = np.array([r["draft"].seconds for r in rs])

        def fixed(baseline: str, *, memory: bool, gpu: bool = False, rs=rs, bench=bench):
            res = [refs[baseline][r["item"]] for r in rs]
            return ([bool(x.score.get(HEADLINE[bench])) for x in res],
                    [price(x.muse_prompt_tokens, x.muse_completion_tokens) for x in res],
                    [x.seconds if gpu and x.route == LOCAL else 0.0 for x in res],
                    [x.seconds for x in res], [x.route != LOCAL for x in res])

        def composed(picked: list[str], probes: bool, rs=rs, draft_s=draft_s):
            chosen = [r["results"][p] for r, p in zip(rs, picked)]
            local = draft_s * (2 if probes else 1)  # draft, and the probes charged the draft's time again
            return ([r["correct"][p] for r, p in zip(rs, picked)],
                    [price(c.muse_prompt_tokens, c.muse_completion_tokens) for c in chosen],
                    local,
                    local + np.array([c.seconds if p != LOCAL else 0.0 for c, p in zip(chosen, picked)]),
                    [p != LOCAL for p in picked])

        direct = refs["muse_direct"]
        runaway = [r["item"] for r in rs if direct[r["item"]].muse_completion_tokens >= OUTPUT_LIMIT]
        out = {"items": len(rs), "memory_build_seconds": build_s, "runaways": len(runaway),
               "muse_direct_cached_prompt_share": muse_cached, "options": []}
        out["options"].append(option("Muse, full context", *fixed("muse_direct", memory=False),
                                     memory=False, build_s=build_s,
                                     note=f"{len(runaway)} answers ran to Muse's output limit"))
        out["muse_direct_usd_per_1k_without_runaways"] = 1000 * float(np.mean(
            [price(direct[r["item"]].muse_prompt_tokens, direct[r["item"]].muse_completion_tokens)
             for r in rs if r["item"] not in runaway]))
        out["options"].append(option("Muse + retrieved raw pages", *fixed("escalate_raw", memory=True),
                                     memory=True, build_s=build_s, note="retrieval, no compression"))
        out["options"].append(option("Muse + team memory", *fixed("escalate_compact", memory=True),
                                     memory=True, build_s=build_s, note="compression, no routing"))
        out["options"].append(option("Rule gate", *fixed("rule_gate", memory=True, gpu=True),
                                     memory=True, build_s=build_s, note="hand-written routing"))
        points = saved["cascade"]["operating_points"]
        picked10 = cascade_choose(rs, points["0.1"][bench]["threshold"])
        n_up = sum(p != LOCAL for p in picked10)
        draws = []
        for _ in range(DRAWS):
            up = set(rng.choice(len(rs), n_up, replace=False).tolist())
            picked = [r["escalate_to"] if i in up else LOCAL for i, r in enumerate(rs)]
            draws.append(option("", *composed(picked, False), memory=True, build_s=build_s))
        out["options"].append({**{k: float(np.mean([d[k] for d in draws])) for k in draws[0]
                                  if k not in ("name", "note")},
                               "name": "Random escalation",
                               "note": "as many questions as the cascade at 10%, mean of 1,000 draws"})
        for budget in ("0.1", "0.2"):
            picked = cascade_choose(rs, points[budget][bench]["threshold"])
            if budget == "0.1":
                assert np.isclose(np.mean([r["correct"][p] for r, p in zip(rs, picked)]),
                                  points[budget][bench]["accuracy"]), bench
            out["options"].append(option(f"Itara cascade, {float(budget):.0%} budget", *composed(picked, True),
                                         memory=True, build_s=build_s,
                                         note="" if budget == "0.1" else "budget chosen after the run"))
        if scores:
            win = [scores[(bench, r["item"])] for r in rs]
            ref_spent = sum(direct[r["item"]].muse_prompt_tokens + direct[r["item"]].muse_completion_tokens
                            for r in rs)
            order = np.argsort(win)[::-1]  # most likely to need Muse first
            draft_wrong = [not r["correct"][LOCAL] for r in rs]
            out["router_auc"] = {"routellm_bert": auc(win, draft_wrong),
                                 "itara_cascade": auc([-r["p_draft"] for r in rs], draft_wrong)}
            for budget in (0.1, 0.2):
                best = None
                for k in range(len(rs) + 1):
                    up = set(order[:k].tolist())
                    picked = [r["escalate_to"] if i in up else LOCAL for i, r in enumerate(rs)]
                    spent = sum(r["prompt"][p] + r["completion"][p] for r, p in zip(rs, picked))
                    right = np.mean([r["correct"][p] for r, p in zip(rs, picked)])
                    if spent <= budget * ref_spent and (best is None or right > best[0]):
                        best = (right, picked)
                picked = best[1]
                chosen = [r["results"][p] for r, p in zip(rs, picked)]
                kept = np.array([p == LOCAL for p in picked])
                out["options"].append(option(
                    f"RouteLLM router, {budget:.0%} budget",
                    [r["correct"][p] for r, p in zip(rs, picked)],
                    [price(c.muse_prompt_tokens, c.muse_completion_tokens) for c in chosen],
                    draft_s * kept, np.where(kept, draft_s, [c.seconds for c in chosen]), ~kept,
                    memory=True, build_s=build_s,
                    note="off the shelf, decides before any draft; threshold chosen on these items"
                         + ("" if budget == 0.1 else "; budget chosen after the run")))
        report["benchmarks"][bench] = out

    path = RESULTS / f"cost_compare_{args.id}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path.relative_to(RESULTS.parent.parent)}")
    for bench, out in report["benchmarks"].items():
        print(f"\n{bench} ({out['items']} items; memory build {out['memory_build_seconds']:.0f} s)")
        print(f"  {'option':34} {'acc':>6} {'esc':>5} {'muse$':>7} {'gpu$':>6} {'$/1k':>7} {'$/1k ok':>8}"
              f" {'med s':>6} {'mean s':>6} {'p90 s':>6}")
        for o in out["options"]:
            print(f"  {o['name']:34} {o['accuracy']:6.1%} {o['escalated']:5.0%} {o['muse_usd_per_1k']:7.2f}"
                  f" {o['gpu_usd_per_1k']:6.2f} {o['usd_per_1k']:7.2f} {o['usd_per_1k_correct']:8.2f}"
                  f" {o['seconds_median']:6.1f} {o['seconds_mean']:6.1f} {o['seconds_p90']:6.1f}")
        if "router_auc" in out:
            print("  AUC for 'Llama's draft is wrong':", {k: round(v, 3) for k, v in out["router_auc"].items()})
    for order in ("related", "random"):
        print(f"\nteam replay, {order} tasks: ingest $ per engineer (shared / alone)")
        for s, a in zip(report["team"][f"{order}_shared"], report["team"][f"{order}_alone"]):
            print(f"  {s['team_size']:>2} engineers: ${s['ingest_usd_per_engineer']:.4f} / ${a['ingest_usd_per_engineer']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", default="full1", help="the run's id")
    parser.add_argument("-n", type=int, default=300, help="items per benchmark, as in the run")
    parser.add_argument("--database", help="the run's database (default: gateway_<id>_full)")
    asyncio.run(main(parser.parse_args()))

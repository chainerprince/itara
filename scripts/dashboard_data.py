"""Write dashboard/public/results.json, the benchmark numbers the dashboard shows, from results/runs/.

    uv run python -m scripts.dashboard_data            # full1, and its minimal-effort rerun

No model is called. results/runs/ is gitignored, so the JSON is committed and the dashboard
works from a clean checkout. Every number comes from the gate files (eval/learn_gate.py),
the replay CSVs (eval/replay.py) and the hand-check sheet (eval/handcheck.py); the same
numbers as results/full_run.ipynb.

- Budgets: for each benchmark and budget, the most accurate option whose Muse tokens fit,
  among the fixed conditions, the per-route gate's lambda sweep and the cascade's threshold
  sweep (the notebook's "Larger budgets" rule). 10% is the north star; 20% and 25% were
  chosen after the run.
- Team scaling: the latest real (not simulated, not stand-in) replay per baseline and order.
- Open models: the same questions with another open model drafting instead of the 8B (OPEN_MODELS,
  all added after the run; same memory, same Muse answers), each against the 8B with paired
  bootstrap intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval.common import RESULTS

ROOT = Path(__file__).parent.parent
OUT = ROOT / "dashboard" / "public" / "results.json"  # the app reads it; the gateway serves it
BENCHES = ("swebench", "locomo")
FIXED = ("llama_only", "rule_gate", "escalate_compact", "escalate_raw")
BUDGETS = (0.10, 0.20, 0.25)
# The open model that drafted in each run. full1 is the full run; each other run repeats only
# its open-model steps, on a copy of full1's memory and with full1's Muse answers
# (scripts/open_model_run.sh), so it compares with full1 item by item. big1 drafted code only.
OPEN_MODELS = {"full1": "Llama 3.1 8B", "big1": "Llama 3.3 70B", "glim1": "Muse Glimmer 30B"}
# Where the cascade sends an escalation: code with summaries, conversations with raw pages.
ESCALATION = {"swebench": "escalate_compact", "locomo": "escalate_raw"}
FAST_MODE = "Itara cascade, 10% budget"  # its name in eval/cost_compare.py's report
# scripts/cache_check.py on the H100 deployment, 29 Oct (results/bringup.ipynb, section 1):
# one 6,150-token team memory asked by three people.
KV_CACHE = [
    {"user": "alice", "team": "web", "seconds": 0.59, "reused_tokens": 0, "note": "first"},
    {"user": "bob", "team": "web", "seconds": 0.43, "reused_tokens": 6144, "note": "teammate"},
    {"user": "carol", "team": "ops", "seconds": 0.53, "reused_tokens": 0, "note": "other team"},
]


def point(row: dict, name: str, kind: str) -> dict:
    return {"name": name, "kind": kind, "accuracy": round(row["accuracy"], 4),
            "share": round(row["muse_token_share"], 5),
            "input_share": round(row["muse_input_share"], 5),
            "kept": round((row.get("routes") or {}).get("local_memory", 0.0), 4)}


def curve(points: list[dict]) -> list[dict]:
    """A sweep as drawn: sorted by tokens, one point per distinct (tokens, accuracy)."""
    seen = {(p["share"], p["accuracy"]): p for p in points}
    return sorted(seen.values(), key=lambda p: (p["share"], p["accuracy"]))


def benchmark(gate: dict, bench: str) -> dict:
    refs = gate["references"][bench]
    fixed = [point(refs[name], name, "fixed") for name in FIXED if name in refs]
    cascade = [point(s[bench], "cascade", "cascade") for s in gate["cascade"]["sweep"]]
    per_route = [point(s[bench], "per_route", "per_route") for s in gate["sweep"]]
    options = fixed + cascade + per_route
    budgets = []
    for budget in BUDGETS:
        fits = [o for o in options if o["share"] <= budget + 1e-9]
        best = max(fits, key=lambda o: (o["accuracy"], -o["share"]))
        budgets.append({"budget": budget, "after_run": budget > 0.10, **best,
                        "gap": round(best["accuracy"] - refs["muse_direct"]["accuracy"], 4)})
    verdict = gate["verdicts"][bench]
    return {
        "muse_direct": {"accuracy": refs["muse_direct"]["accuracy"],
                        "tokens": refs["muse_direct"]["muse_prompt_tokens"]
                        + refs["muse_direct"]["muse_completion_tokens"]},
        "oracle": point(refs["oracle"], "oracle", "oracle"),
        "fixed": fixed,
        "cascade": curve(cascade),
        "per_route": curve(per_route),
        "north_star": {**point(gate["cascade"]["operating_points"]["0.1"][bench], "cascade", "cascade"),
                       "pass": verdict["pass"]},
        "budgets": budgets,
        "auc": round(gate["signal"][bench]["gate_auc"], 3),
    }


def scaling() -> dict:
    latest = {}
    for path in sorted(RESULTS.glob("replay_*.csv")):  # names end in a timestamp
        rows = list(csv.DictReader(path.open()))
        if rows and rows[0]["simulated"] == "False" and rows[0]["muse_is_standin"] == "False":
            latest[rows[0]["baseline"], rows[0]["order"]] = rows
    out = {}
    for (baseline, order), rows in latest.items():
        series = "shared" if baseline == "rule_gate" else "no_sharing"
        out.setdefault(order, {})[series] = [
            {"team_size": int(r["team_size"]), "accuracy": float(r["accuracy"]),
             "pages_per_user": round(float(r["ingest_pages_per_user"]), 2),
             "compute_tokens_per_user": round(float(r["compute_tokens_per_user"]), 1)} for r in rows]
    return out


def raw(run_id: str, bench: str, baseline: str) -> dict[str, dict]:
    path = RESULTS / "raw" / f"{bench}_{baseline}_{run_id}-{bench}-{baseline}.jsonl"
    return {r["item_id"]: r for r in map(json.loads, path.open())}


def right(run_id: str, bench: str, baseline: str) -> dict[str, bool]:
    key = "hit" if bench == "swebench" else "correct"
    return {i: bool(r["score"][key]) for i, r in raw(run_id, bench, baseline).items()}


def perfect_check(base: str, run_id: str, bench: str) -> dict:
    """Escalate exactly the drafts that are wrong (by the cascade's route): the best any gate
    that judges only the draft could do, and what it would cost."""
    draft = right(run_id, bench, "llama_only")
    escalated, direct = raw(base, bench, ESCALATION[bench]), raw(base, bench, "muse_direct")
    key = "hit" if bench == "swebench" else "correct"
    tokens = lambda r: r["muse_prompt_tokens"] + r["muse_completion_tokens"]  # noqa: E731
    wrong = [i for i in draft if not draft[i]]
    return {"wrong_drafts": len(wrong) / len(draft),
            "accuracy": sum(draft[i] or bool(escalated[i]["score"][key]) for i in draft) / len(draft),
            "share": round(sum(tokens(escalated[i]) for i in wrong) / sum(tokens(direct[i]) for i in draft), 5)}


def gains(base: str, run_id: str, bench: str, resamples: int = 4000, seed: int = 0) -> dict:
    """run_id's accuracy minus base's on the same items, alone and in the cascade at 10%, with
    paired bootstrap intervals (a generator per pair, so adding a model changes no other's)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, baseline in (("alone", "llama_only"), ("cascade_10", "full")):
        a, b = right(base, bench, baseline), right(run_id, bench, baseline)
        ids = sorted(set(a) & set(b))
        x, y = (np.array([d[i] for i in ids], float) for d in (a, b))
        idx = rng.integers(0, len(ids), (resamples, len(ids)))
        diff = y[idx].mean(axis=1) - x[idx].mean(axis=1)
        out[name] = {"points": float(y.mean() - x.mean()),
                     "interval": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                     "p_better": float((diff > 0).mean())}
    return out


def open_models(base: str) -> dict:
    """For each benchmark, every open model that drafted it: alone, and in the cascade at 10%
    and 20% of Muse's tokens, with its gain over base's model on the same items."""
    out = {}
    for bench in BENCHES:
        models = []
        for run_id, label in OPEN_MODELS.items():
            path = RESULTS / f"gate_{run_id}.json"
            if not path.exists():
                continue
            gate = json.loads(path.read_text())
            if bench not in gate["references"]:
                continue
            refs, points = gate["references"][bench], gate["cascade"]["operating_points"]
            cost = json.loads((RESULTS / f"cost_compare_{run_id}.json").read_text())
            fast = next(o for o in cost["benchmarks"][bench]["options"] if o["name"] == FAST_MODE)
            drafts = json.loads((RESULTS / f"{bench}_llama_only_{run_id}-{bench}-llama_only.json").read_text())
            model = {"run_id": run_id, "label": label, "gpus": cost["gpus"],
                     "reasoning": drafts.get("llama_reasoning_strength", ""),  # "": doesn't reason
                     # Muse and GPU dollars per 1,000 questions, and the median wait (eval/cost_compare.py)
                     "fast_mode": {"usd_per_1k": round(fast["usd_per_1k"], 2),
                                   "seconds_median": round(fast["seconds_median"], 1)},
                     "alone": refs["llama_only"]["accuracy"],
                     "cascade": {b: {"accuracy": points[b][bench]["accuracy"],
                                     "share": round(points[b][bench]["muse_token_share"], 5),
                                     "kept": points[b][bench]["routes"].get("local_memory", 0.0)}
                                 for b in ("0.1", "0.2")},
                     "auc": round(gate["cascade"]["auc"][bench], 3),
                     "perfect_check": perfect_check(base, run_id, bench)}
            if run_id != base:
                model["gain"] = gains(base, run_id, bench)
            models.append(model)
        if len(models) > 1:
            muse = json.loads((RESULTS / f"gate_{base}.json").read_text())["references"][bench]
            out[bench] = {"muse_direct": muse["muse_direct"]["accuracy"], "models": models}
    return out


def handcheck(run_id: str) -> dict:
    rows = [r for r in csv.DictReader((RESULTS / f"handcheck_locomo_{run_id}.csv").open())
            if r["human"]]
    return {"checked": len(rows), "agree": sum(r["human"] == r["judge"] for r in rows)}


def main(args) -> None:
    efforts = {}
    for effort, run_id in (("default", args.id), ("minimal", f"{args.id}-minimal")):
        path = RESULTS / f"gate_{run_id}.json"
        if not path.exists():
            continue
        gate = json.loads(path.read_text())
        efforts[effort] = {"run_id": run_id, "after_run": effort != "default",
                           "benchmarks": {b: benchmark(gate, b) for b in BENCHES}}
    data = {"run_id": args.id, "items_per_benchmark": 300, "efforts": efforts, "scaling": scaling(), "kv_cache": KV_CACHE,
            "handcheck": handcheck(args.id), "open_models": open_models(args.id)}
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
    for effort, e in efforts.items():
        for bench, b in e["benchmarks"].items():
            print(f"  {effort:8}{bench:9} muse_direct {b['muse_direct']['accuracy']:.1%}  " + "  ".join(
                f"{x['budget']:.0%}: {x['accuracy']:.1%} ({x['name']}, {x['share']:.1%})"
                for x in b["budgets"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", default="full1", help="the full run's id")
    main(parser.parse_args())

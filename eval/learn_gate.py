"""Train the learned gate on escalation regret, replay it offline, and mark the north star.

    uv run python -m eval.learn_gate --id full1 -n 300

Needs, on the same items, per-item results of the routes the gate chooses between
(llama_only = local_memory, escalate_compact, and escalate_raw if it was run) plus
muse_direct (the ceiling) and rule_gate: scripts/full_run.sh, then scripts/gate_run.sh,
which adds escalate_raw and Llama's confidence (eval/confidence.py) and runs this. No model
is called here.

- Features (gateway/gate.py) come from the question, the memory the gateway retrieves for
  it (searched again in the run's database: search is deterministic) and Llama's
  llama_only answer, which is exactly the cascade's draft. If eval/confidence.py has run,
  Llama's confidence in that draft is added (logprobs, resamples, a self-check).
- One logistic model per route predicts whether that route answers right. Predictions are
  out of fold (5 folds grouped by team, so a conversation or repo is never in both training
  and test), so every replayed number is on items the gate didn't train on.
- For each lam the gate picks argmax P(right) - lam * estimated Muse dollars, and the replay
  scores that choice with the answer and tokens the route actually produced for the item
  (answers are at temperature 0, so the live gate would get the same ones).
- Operating points. Per benchmark: the most accurate lam whose Muse tokens are at most 10% of
  that benchmark's muse_direct (a team sets lam against its own budget). Pooled: the same
  over both benchmarks at once, kept for reference.
- The cascade (gateway.gate.CascadeGate), the gate the gateway ships: one logistic model
  scores P(draft right) from Llama's confidence alone (all features if eval/confidence.py
  hasn't run), out of fold on the same folds. Drafts under a threshold escalate: code memory
  with summaries, other memory with raw pages. The threshold is swept like lam. Its
  per-benchmark points at --budget give the verdicts and the `full` baselines; points at
  --budgets are reported too. The saved gate is trained on every item, with the threshold
  of --live-group's point at --live-budget (default: code at 20%, for a team that values
  accuracy over the last tokens; pooled budgets of 20% and up just escalate everything).
  GATE_THRESHOLD overrides it in the gateway: the printed table maps budgets to thresholds.

North star, per benchmark: accuracy within 3 points of muse_direct, with at most 10% of its
Muse tokens. Input-only token shares are reported too, since Muse's reasoning makes every
escalation cost ~1,000 output tokens however short the prompt.

Also reports how well each feature, and the gate's out-of-fold P(draft right), tells right
drafts from wrong ones (AUC: 0.5 is chance, 1 is perfect).

Writes results/runs/gate_<id>.json (both gates' sweeps, references, verdicts, signal),
gate_<id>_sweep.csv (the per-route gate), gate_<id>_model.json (the cascade: the GATE_PATH
file), gate_<id>_routes_model.json (the per-route gate) and a `full` baseline per benchmark
(the cascade's offline replay at --budget) for the results notebooks.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import replace

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from eval import locomo, swebench_loc
from eval.common import RESULTS, Result, RunLog, summarise
from eval.confidence import load_confidence
from gateway.accounting import Ledger
from gateway.config import Settings
from gateway.gate import (CONFIDENCE_FEATURES, FEATURES, CascadeGate, LearnedGate, Route,
                          escalation_route, estimated_prompt_tokens, gate_features)
from memory.store import MemoryStore, Scope, default_embedder

BENCHES = ("swebench", "locomo")
HEADLINE = {"swebench": "hit", "locomo": "correct"}
ROUTE_BASELINE = {Route.LOCAL_MEMORY: "llama_only", Route.ESCALATE_COMPACT: "escalate_compact",
                  Route.ESCALATE_RAW: "escalate_raw"}
REFERENCES = ("muse_direct", "rule_gate", "llama_only", "escalate_compact", "escalate_raw")
LAMBDAS = [0.0, *np.geomspace(1, 1e5, 161).tolist(), 1e12]  # 1e12: never worth escalating
FOLDS = 5
NO_PROBE = {"llama_prompt_tokens": 0, "llama_completion_tokens": 0, "cached_tokens": 0,
            "compute_tokens": 0}


def run_results(run_id: str, bench: str, baseline: str) -> dict[str, Result]:
    log = RunLog(f"{bench}_{baseline}", f"{run_id}-{bench}-{baseline}")
    return {r.item_id: r for r in log.results()}


def muse_tokens(r: Result) -> int:
    return r.muse_prompt_tokens + r.muse_completion_tokens


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method; the last weight is the intercept."""
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    if y.all() or not y.any():  # one class: a constant, kept off infinity
        w[-1] = np.log((y.mean() * len(y) + 0.5) / ((1 - y.mean()) * len(y) + 0.5))
        return w
    reg = l2 * np.eye(len(w))
    reg[-1, -1] = 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-(xb @ w)))
        gradient = xb.T @ (p - y) + reg @ w
        hessian = xb.T @ (xb * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """P(a random right draft scores above a random wrong one); ties count half."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    s, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float((s > n).mean() + 0.5 * (s == n).mean())


def folds(groups: list[str], k: int = FOLDS) -> np.ndarray:
    """Fold per row: whole groups, biggest first into the smallest fold (deterministic)."""
    sizes: dict[str, int] = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1
    load, fold_of = [0] * k, {}
    for g in sorted(sizes, key=lambda g: (-sizes[g], g)):
        fold_of[g] = load.index(min(load))
        load[fold_of[g]] += sizes[g]
    return np.array([fold_of[g] for g in groups])


def train(rows: list[dict], routes: list[str], lam: float, features: list[str]) -> LearnedGate:
    x = np.array([[r["features"][f] for f in features] for r in rows])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    weights = {route: fit_logistic(z, np.array([r["correct"][route] for r in rows], float)).tolist()
               for route in routes}
    escalated = [r["completion"][route] for r in rows for route in routes[1:]]
    return LearnedGate(routes=routes, mean=mean.tolist(), scale=scale.tolist(), weights=weights,
                       output_tokens=float(np.mean(escalated)) if escalated else 0.0, lam=lam,
                       features=features)


def out_of_fold(rows: list[dict], routes: list[str], price, features: list[str]) -> None:
    """Adds each row's out-of-fold P(right) and estimated cost per route."""
    fold = folds([r["team"] for r in rows])
    for k in range(FOLDS):
        test = [r for r, f in zip(rows, fold) if f == k]
        if not test:
            continue
        gate = train([r for r, f in zip(rows, fold) if f != k], routes, 0.0, features)
        for r in test:
            r["p"] = gate.probabilities(r["features"])
            r["cost"] = gate.costs(r["prompt_estimate"], price)


def choose(rows: list[dict], routes: list[str], lam: float) -> list[str]:
    gate = LearnedGate(routes, [], [], {}, 0.0, lam)
    return [gate.decide(r["p"], r["cost"]).value for r in rows]


def score(rows: list[dict], picked: list[str]) -> dict:
    n = len(rows) or 1
    prompt = sum(r["prompt"][p] for r, p in zip(rows, picked))
    completion = sum(r["completion"][p] for r, p in zip(rows, picked))
    shares = {route: picked.count(route) / n for route in sorted(set(picked))}
    return {"items": len(rows), "accuracy": sum(r["correct"][p] for r, p in zip(rows, picked)) / n,
            "muse_prompt_tokens": prompt, "muse_completion_tokens": completion,
            "muse_usd": sum(r["usd"][p] for r, p in zip(rows, picked)),
            "llama_compute_tokens": sum(r["draft"].compute_tokens + r["probe"]["compute_tokens"]
                                        for r in rows),
            "routes": shares}


def reference(rows: list[dict], refs: dict) -> dict:
    out = {}
    for baseline in REFERENCES:
        if not all(r["item"] in refs[baseline] for r in rows):
            continue
        rs = [refs[baseline][r["item"]] for r in rows]
        n = len(rs) or 1
        out[baseline] = {"items": len(rs),
                         "accuracy": sum(bool(x.score.get(r["headline"])) for x, r in zip(rs, rows)) / n,
                         "muse_prompt_tokens": sum(x.muse_prompt_tokens for x in rs),
                         "muse_completion_tokens": sum(x.muse_completion_tokens for x in rs)}
    # The oracle: the cheapest route that's right, if any is. An upper bound for any gate.
    oracle = [next((route for route in r["routes"] if r["correct"][route]), r["routes"][0])
              for r in rows]
    out["oracle"] = score(rows, oracle)
    return out


def with_ratios(point: dict, ceiling: dict) -> dict:
    total = ceiling["muse_prompt_tokens"] + ceiling["muse_completion_tokens"]
    tokens = point["muse_prompt_tokens"] + point["muse_completion_tokens"]
    return {**point, "muse_token_share": tokens / total if total else None,
            "muse_input_share": (point["muse_prompt_tokens"] / ceiling["muse_prompt_tokens"]
                                 if ceiling["muse_prompt_tokens"] else None),
            "accuracy_gap": point["accuracy"] - ceiling["accuracy"]}


def signal(rows: list[dict], features: list[str]) -> dict:
    """How well each feature, and the gate's P(draft right), separate right drafts from wrong."""
    right = [r["correct"][Route.LOCAL_MEMORY] for r in rows]
    return {"gate_auc": auc([r["p"][Route.LOCAL_MEMORY] for r in rows], right),
            "draft_right": sum(right) / len(right),
            "feature_auc": {f: auc([r["features"][f] for r in rows], right) for f in features}}


def operating_point(sweep: list[dict], group: str, budget: float) -> dict:
    """The most accurate lam within budget for a group (ties: the cheaper, larger lam)."""
    within = [p for p in sweep if group in p and p[group]["muse_token_share"] <= budget]
    return max(within or sweep[-1:], key=lambda p: (p[group]["accuracy"], p["lam"]))


def train_cascade(rows: list[dict], features: list[str], threshold: float) -> CascadeGate:
    x = np.array([[r["features"][f] for f in features] for r in rows])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    y = np.array([r["correct"][Route.LOCAL_MEMORY] for r in rows], float)
    return CascadeGate(mean=mean.tolist(), scale=scale.tolist(),
                       weights=fit_logistic((x - mean) / scale, y).tolist(),
                       threshold=threshold, features=features)


def cascade_out_of_fold(rows: list[dict], features: list[str]) -> None:
    """Adds each row's out-of-fold P(draft right) under the cascade, as `p_draft`."""
    fold = folds([r["team"] for r in rows])
    for k in range(FOLDS):
        test = [r for r, f in zip(rows, fold) if f == k]
        if test:
            gate = train_cascade([r for r, f in zip(rows, fold) if f != k], features, 0.0)
            for r in test:
                r["p_draft"] = gate.p_right(r["features"])


def cascade_choose(rows: list[dict], threshold: float) -> list[str]:
    return [Route.LOCAL_MEMORY.value if r["p_draft"] >= threshold else r["escalate_to"]
            for r in rows]


def cascade_point(sweep: list[dict], group: str, budget: float) -> dict:
    """The most accurate threshold within budget (ties: the cheaper, lower threshold)."""
    within = [p for p in sweep if p[group]["muse_token_share"] <= budget]
    return max(within or sweep[:1], key=lambda p: (p[group]["accuracy"], -p["threshold"]))


async def load_rows(args, routes: list[str], price) -> tuple[list[dict], dict, int, bool]:
    """One row per item that every needed baseline answered; features from the run's memory."""
    refs = {b: {} for b in REFERENCES}
    rows, mismatched = [], 0
    probes = {b: load_confidence(args.id, b) for b in BENCHES}
    confident = not args.no_confidence and all(probes.values())
    url = args.database or f"{Settings().database_url.rsplit('/', 1)[0]}/gateway_{args.id.replace('-', '_')}_full"
    async with AsyncConnectionPool(url, configure=register_vector_async) as pool:
        store = MemoryStore(pool, default_embedder())
        for bench in BENCHES:
            items = (swebench_loc.load(args.n, args.seed) if bench == "swebench"
                     else locomo.load(args.n, args.seed))
            results = {b: run_results(args.id, bench, b) for b in set(REFERENCES) | {"llama_only"}}
            for b in REFERENCES:
                refs[b].update(results[b])
            for item in items:
                by_route = {route: results[ROUTE_BASELINE[route]].get(item.id) for route in routes}
                if any(r is None for r in by_route.values()) or item.id not in results["muse_direct"]:
                    continue
                probe = probes[bench].get(item.id)
                if confident and probe is None:
                    continue
                draft = by_route[Route.LOCAL_MEMORY]
                scope = Scope(item.team_id, "learn-gate")
                hits = await store.search(scope, item.question)
                mismatched += [h.source or "" for h in hits] != draft.retrieved_sources
                pages = await store.pages(scope, [h.id for h in hits])
                vectors = await store.embed([item.question, draft.answer, *(h.content for h in hits)])
                features = gate_features(item.question, hits, draft.answer, vectors)
                if confident:
                    features |= probe["features"]
                rows.append({
                    "item": item.id, "bench": bench, "team": item.team_id, "routes": routes,
                    "headline": HEADLINE[bench], "draft": draft, "results": by_route,
                    "features": features,
                    "probe": ({k: probe[k] for k in NO_PROBE} if confident else NO_PROBE),
                    "prompt_estimate": estimated_prompt_tokens(item.question, hits, pages),
                    "escalate_to": (escalation_route(hits) if Route.ESCALATE_RAW in routes
                                    else Route.ESCALATE_COMPACT).value,
                    "correct": {route: bool(r.score.get(HEADLINE[bench])) for route, r in by_route.items()},
                    "prompt": {route: r.muse_prompt_tokens for route, r in by_route.items()},
                    "completion": {route: r.muse_completion_tokens for route, r in by_route.items()},
                    "usd": {route: price(r.muse_prompt_tokens, r.muse_completion_tokens)
                            for route, r in by_route.items()},
                })
    return rows, refs, mismatched, confident


def write_full_baseline(args, bench: str, rows: list[dict], picked: list[str], gate: dict) -> None:
    """The offline replay as a `full` run, so the results notebooks can show it."""
    run_id = f"{args.id}-{bench}-full"
    log = RunLog(f"{bench}_full", run_id)
    log.path.unlink(missing_ok=True)
    results = []
    for r, route in zip(rows, picked):
        draft, chosen, probe = r["draft"], r["results"][route], r["probe"]
        # Local work is the draft plus the confidence calls, whichever route answers.
        result = replace(chosen, baseline="full", route=route, run_id=run_id,
                         llama_prompt_tokens=draft.llama_prompt_tokens + probe["llama_prompt_tokens"],
                         llama_completion_tokens=(draft.llama_completion_tokens
                                                  + probe["llama_completion_tokens"]),
                         cached_tokens=draft.cached_tokens + probe["cached_tokens"],
                         compute_tokens=draft.compute_tokens + probe["compute_tokens"],
                         seconds=draft.seconds + (chosen.seconds if route != Route.LOCAL_MEMORY else 0))
        log.append(result)
        results.append(result)
    # Labels from a run that called Muse, so Muse's settings (its reasoning effort) carry over.
    labels_path = RESULTS / f"{bench}_escalate_compact_{args.id}-{bench}-escalate_compact.json"
    labels = {k: v for k, v in json.loads(labels_path.read_text()).items()
              if k in ("llama_model", "muse_model", "muse_is_standin", "muse_reasoning_effort",
                       "simulated", "ingest_concurrency", "answer_temperature")}
    log.write_summary({"bench": bench, "baseline": "full", "seed": args.seed, **labels,
                       "offline_replay": True, **gate, **summarise(results, HEADLINE[bench])})


async def main(args) -> None:
    settings = Settings()
    price = Ledger(None, 0.0, settings.muse_input_price_per_m, settings.muse_output_price_per_m).cost
    routes = [Route.LOCAL_MEMORY.value, Route.ESCALATE_COMPACT.value]
    if all(run_results(args.id, b, "escalate_raw") for b in BENCHES):
        routes.append(Route.ESCALATE_RAW.value)
    rows, refs, mismatched, confident = await load_rows(args, routes, price)
    if not rows:
        raise SystemExit(f"No item has results for every route {routes} and muse_direct "
                         f"under run id {args.id!r}")
    features = list(FEATURES) + (list(CONFIDENCE_FEATURES) if confident else [])
    out_of_fold(rows, routes, price, features)
    cascade_features = list(CONFIDENCE_FEATURES) if confident else list(FEATURES)
    cascade_out_of_fold(rows, cascade_features)

    groups = {"all": rows, **{b: [r for r in rows if r["bench"] == b] for b in BENCHES}}
    groups = {g: rs for g, rs in groups.items() if rs}
    ceilings = {g: reference(rs, refs)["muse_direct"] for g, rs in groups.items()}

    def replay(picked: list[str], key: str, value: float) -> dict:
        point = {key: value}
        for g, rs in groups.items():
            chosen = picked if g == "all" else [p for r, p in zip(rows, picked) if r["bench"] == g]
            point[g] = with_ratios(score(rs, chosen), ceilings[g])
        return point

    # The per-route gate (LearnedGate), kept for comparison.
    sweep = [replay(choose(rows, routes, lam), "lam", lam) for lam in LAMBDAS]
    pooled = operating_point(sweep, "all", args.budget)
    points = {b: operating_point(sweep, b, args.budget) for b in BENCHES if b in groups}
    train(rows, routes, pooled["lam"], features).save(
        RESULTS / f"gate_{args.id}_routes_model.json", trained_on=args.id, items=len(rows),
        budget=args.budget, lam_by_bench={b: p["lam"] for b, p in points.items()})

    # The cascade: the gateway's gate.
    thresholds = [0.0, *sorted({r["p_draft"] for r in rows}), 1.01]  # 1.01: always escalate
    cascade_sweep = [replay(cascade_choose(rows, t), "threshold", t) for t in thresholds]
    budgets = sorted({args.budget, *args.budgets})
    cascade_points = {}  # budget -> group -> the most accurate point within it
    for budget in budgets:
        cascade_points[f"{budget:g}"] = {}
        for g in groups:
            p = cascade_point(cascade_sweep, g, budget)
            cascade_points[f"{budget:g}"][g] = {"threshold": p["threshold"], **p[g]}
    live = cascade_point(cascade_sweep, args.live_group, args.live_budget)
    model_path = RESULTS / f"gate_{args.id}_model.json"
    train_cascade(rows, cascade_features, live["threshold"]).save(
        model_path, trained_on=args.id, items=len(rows), live_budget=args.live_budget,
        live_group=args.live_group)

    verdicts = {}
    for bench in BENCHES:
        if bench not in groups:
            continue
        q = cascade_points[f"{args.budget:g}"][bench]
        picked = cascade_choose(groups[bench], q["threshold"])
        write_full_baseline(args, bench, groups[bench], picked,
                            {"gate": "cascade", "threshold": q["threshold"]})
        accuracy_ok, tokens_ok = q["accuracy_gap"] >= -args.margin, q["muse_token_share"] <= args.budget
        verdicts[bench] = {"threshold": q["threshold"], "accuracy_ok": accuracy_ok,
                           "tokens_ok": tokens_ok, "pass": accuracy_ok and tokens_ok}
    signals = {g: signal(rs, features) for g, rs in groups.items()}
    cascade_auc = {g: auc([r["p_draft"] for r in rs], [r["correct"][Route.LOCAL_MEMORY] for r in rs])
                   for g, rs in groups.items()}
    summary = {"run_id": args.id, "routes": routes, "features": features, "confidence": confident,
               "items": len(rows), "retrieval_mismatches": mismatched, "budget": args.budget,
               "margin": args.margin, "gate": "cascade", "verdicts": verdicts,
               "cascade": {"features": cascade_features, "auc": cascade_auc,
                           "live": {"budget": args.live_budget, "group": args.live_group,
                                    "threshold": live["threshold"], **{g: live[g] for g in groups}},
                           "operating_points": cascade_points, "sweep": cascade_sweep},
               # The per-route gate:
               "operating_points": {b: {"lam": p["lam"], **p[b]} for b, p in points.items()},
               "operating_point": pooled, "signal": signals,
               "references": {g: {k: with_ratios(v, ceilings[g]) for k, v in reference(rs, refs).items()}
                              for g, rs in groups.items()},
               "sweep": sweep}
    (RESULTS / f"gate_{args.id}.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (RESULTS / f"gate_{args.id}_sweep.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lam", "group", "accuracy", "muse_token_share", "muse_input_share",
                         "muse_usd", *(f"share_{r}" for r in routes)])
        for p in sweep:
            for g in groups:
                q = p[g]
                writer.writerow([p["lam"], g, q["accuracy"], q["muse_token_share"],
                                 q["muse_input_share"], q["muse_usd"],
                                 *(q["routes"].get(r, 0.0) for r in routes)])

    print(f"{len(rows)} items, routes {routes}; {mismatched} retrievals differ from the run's; "
          f"{'with' if confident else 'WITHOUT'} Llama's confidence ({len(features)} features)")
    print("\nHow well each gate tells a right draft from a wrong one (AUC, 0.5 = chance):")
    for g in BENCHES:
        if g in signals:
            top = sorted(((a, f) for f, a in signals[g]["feature_auc"].items() if a is not None),
                         key=lambda t: -abs(t[0] - 0.5))[:4]
            print(f"  {g:9} cascade {cascade_auc[g]:.3f}, per-route gate {signals[g]['gate_auc']:.3f}; "
                  "strongest features: " + ", ".join(f"{f} {a:.2f}" for a, f in top))
    print(f"\n{'':26}{'accuracy':>10}{'tokens':>9}{'input':>8}   (shares of muse_direct's Muse tokens)")
    for g in BENCHES:
        if g not in points:
            continue
        print(f"-- {g}")
        lines = [*summary["references"][g].items(), (f"per-route gate, {args.budget:.0%}", points[g][g])]
        lines += [(f"cascade, {b:.0%}", cascade_points[f"{b:g}"][g]) for b in budgets]
        for name, point in lines:
            print(f"{name:26}{point['accuracy']:>10.3f}{point['muse_token_share']:>9.1%}"
                  f"{point['muse_input_share'] or 0:>8.1%}")
    print("\nCascade thresholds by budget (GATE_THRESHOLD; out of fold):")
    for budget in budgets:
        print(f"  {budget:>4.0%}: " + "; ".join(
            f"{g} {q['threshold']:.3f} ({q['accuracy']:.3f}, keeps {q['routes'].get('local_memory', 0):.0%})"
            for g, q in cascade_points[f"{budget:g}"].items()))
    print(f"Saved cascade: threshold {live['threshold']:.3f} ({args.live_group} at {args.live_budget:.0%}): "
          + ", ".join(f"{g} {live[g]['accuracy']:.3f} at {live[g]['muse_token_share']:.1%}"
                      for g in BENCHES if g in live))
    for bench, v in verdicts.items():
        q = cascade_points[f"{args.budget:g}"][bench]
        print(f"{'PASS' if v['pass'] else 'FAIL'} {bench} (cascade, threshold {v['threshold']:.3f}, routes "
              f"{ {k: round(x, 2) for k, x in q['routes'].items()} }): accuracy {q['accuracy']:.3f} "
              f"vs muse_direct {ceilings[bench]['accuracy']:.3f} (gap {q['accuracy_gap']:+.3f}, "
              f"allowed -{args.margin}), Muse tokens {q['muse_token_share']:.1%} "
              f"(allowed {args.budget:.0%})")
    print(f"\nWrote {RESULTS / f'gate_{args.id}.json'}, the sweep CSV and {model_path}"
          f" (set GATE_PATH={model_path} to use it in the gateway)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="the FULL_ID of scripts/full_run.sh")
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--database", help="default: gateway_<id>_full on DATABASE_URL's server")
    parser.add_argument("--budget", type=float, default=0.10,
                        help="Muse tokens allowed, as a share of muse_direct's")
    parser.add_argument("--margin", type=float, default=0.03,
                        help="accuracy points below muse_direct still counted as a match")
    parser.add_argument("--budgets", type=float, nargs="*", default=[0.20, 0.25],
                        help="more budgets to report the cascade at (chosen after the full run)")
    parser.add_argument("--live-budget", type=float, default=0.20,
                        help="the budget the saved cascade's threshold is set for")
    parser.add_argument("--live-group", default="swebench", choices=["all", *BENCHES],
                        help="the benchmark that budget applies to")
    parser.add_argument("--no-confidence", action="store_true",
                        help="leave out Llama's confidence even if eval/confidence.py has run")
    asyncio.run(main(parser.parse_args()))

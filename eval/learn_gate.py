"""Train the learned gate on escalation regret, replay it offline, and mark the north star.

    uv run python -m eval.learn_gate --id full1 -n 300

Needs, on the same items, per-item results of the routes the gate chooses between
(llama_only = local_memory, escalate_compact, and escalate_raw if it was run) plus
muse_direct (the ceiling) and rule_gate: scripts/full_run.sh, then scripts/gate_run.sh,
which adds escalate_raw and runs this. No model is called here.

- Features (gateway/gate.py) come from the question, the memory the gateway retrieves for
  it (searched again in the run's database: search is deterministic) and Llama's
  llama_only answer, which is exactly the cascade's draft.
- One logistic model per route predicts whether that route answers right. Predictions are
  out of fold (5 folds grouped by team, so a conversation or repo is never in both training
  and test), so every replayed number is on items the gate didn't train on.
- For each lam the gate picks argmax P(right) - lam * estimated Muse dollars, and the replay
  scores that choice with the answer and tokens the route actually produced for the item
  (answers are at temperature 0, so the live gate would get the same ones).
- Operating point: the most accurate lam whose Muse tokens (prompt + completion, pooled
  over both benchmarks) are at most 10% of muse_direct's. The gate saved for the live
  gateway (GATE_PATH) is trained on every item, with that lam.

North star, per benchmark: accuracy within 3 points of muse_direct, with at most 10% of its
Muse tokens. Input-only token shares are reported too, since Muse's reasoning makes every
escalation cost ~1,000 output tokens however short the prompt.

Writes eval/results/gate_<id>.json (sweep, references, verdicts), gate_<id>_sweep.csv,
gate_<id>_model.json (the GATE_PATH file), and a `full` baseline per benchmark (the offline
replay at the operating point) for eval/day6_results.ipynb.
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
from gateway.accounting import Ledger
from gateway.config import Settings
from gateway.gate import FEATURES, LearnedGate, Route, estimated_prompt_tokens, gate_features
from memory.store import MemoryStore, Scope, default_embedder

BENCHES = ("swebench", "locomo")
HEADLINE = {"swebench": "hit", "locomo": "correct"}
ROUTE_BASELINE = {Route.LOCAL_MEMORY: "llama_only", Route.ESCALATE_COMPACT: "escalate_compact",
                  Route.ESCALATE_RAW: "escalate_raw"}
REFERENCES = ("muse_direct", "rule_gate", "llama_only", "escalate_compact", "escalate_raw")
LAMBDAS = [0.0, *np.geomspace(1, 1e5, 41).tolist(), 1e12]  # 1e12: never worth escalating
FOLDS = 5


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


def train(rows: list[dict], routes: list[str], lam: float) -> LearnedGate:
    x = np.array([[r["features"][f] for f in FEATURES] for r in rows])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    weights = {route: fit_logistic(z, np.array([r["correct"][route] for r in rows], float)).tolist()
               for route in routes}
    escalated = [r["completion"][route] for r in rows for route in routes[1:]]
    return LearnedGate(routes=routes, mean=mean.tolist(), scale=scale.tolist(), weights=weights,
                       output_tokens=float(np.mean(escalated)) if escalated else 0.0, lam=lam)


def out_of_fold(rows: list[dict], routes: list[str], price) -> None:
    """Adds each row's out-of-fold P(right) and estimated cost per route."""
    fold = folds([r["team"] for r in rows])
    for k in range(FOLDS):
        test = [r for r, f in zip(rows, fold) if f == k]
        if not test:
            continue
        gate = train([r for r, f in zip(rows, fold) if f != k], routes, lam=0.0)
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
            "llama_compute_tokens": sum(r["draft"].compute_tokens for r in rows), "routes": shares}


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


async def load_rows(args, routes: list[str], price) -> tuple[list[dict], dict, int]:
    """One row per item that every needed baseline answered; features from the run's memory."""
    refs = {b: {} for b in REFERENCES}
    rows, mismatched = [], 0
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
                draft = by_route[Route.LOCAL_MEMORY]
                scope = Scope(item.team_id, "learn-gate")
                hits = await store.search(scope, item.question)
                mismatched += [h.source or "" for h in hits] != draft.retrieved_sources
                pages = await store.pages(scope, [h.id for h in hits])
                vectors = await store.embed([item.question, draft.answer, *(h.content for h in hits)])
                rows.append({
                    "item": item.id, "bench": bench, "team": item.team_id, "routes": routes,
                    "headline": HEADLINE[bench], "draft": draft, "results": by_route,
                    "features": gate_features(item.question, hits, draft.answer, vectors),
                    "prompt_estimate": estimated_prompt_tokens(item.question, hits, pages),
                    "correct": {route: bool(r.score.get(HEADLINE[bench])) for route, r in by_route.items()},
                    "prompt": {route: r.muse_prompt_tokens for route, r in by_route.items()},
                    "completion": {route: r.muse_completion_tokens for route, r in by_route.items()},
                    "usd": {route: price(r.muse_prompt_tokens, r.muse_completion_tokens)
                            for route, r in by_route.items()},
                })
    return rows, refs, mismatched


def write_full_baseline(args, bench: str, rows: list[dict], picked: list[str], lam: float) -> None:
    """The offline replay as a `full` run, so the results notebook can show it."""
    run_id = f"{args.id}-{bench}-full"
    log = RunLog(f"{bench}_full", run_id)
    log.path.unlink(missing_ok=True)
    results = []
    for r, route in zip(rows, picked):
        draft, chosen = r["draft"], r["results"][route]
        result = replace(chosen, baseline="full", route=route, run_id=run_id,
                         llama_prompt_tokens=draft.llama_prompt_tokens,
                         llama_completion_tokens=draft.llama_completion_tokens,
                         cached_tokens=draft.cached_tokens, compute_tokens=draft.compute_tokens,
                         seconds=draft.seconds + (chosen.seconds if route != Route.LOCAL_MEMORY else 0))
        log.append(result)
        results.append(result)
    labels_path = RESULTS / f"{bench}_llama_only_{args.id}-{bench}-llama_only.json"
    labels = {k: v for k, v in json.loads(labels_path.read_text()).items()
              if k in ("llama_model", "muse_model", "muse_is_standin", "simulated",
                       "ingest_concurrency", "answer_temperature")}
    log.write_summary({"bench": bench, "baseline": "full", "seed": args.seed, **labels,
                       "offline_replay": True, "lam": lam, **summarise(results, HEADLINE[bench])})


async def main(args) -> None:
    settings = Settings()
    price = Ledger(None, 0.0, settings.muse_input_price_per_m, settings.muse_output_price_per_m).cost
    routes = [Route.LOCAL_MEMORY.value, Route.ESCALATE_COMPACT.value]
    if all(run_results(args.id, b, "escalate_raw") for b in BENCHES):
        routes.append(Route.ESCALATE_RAW.value)
    rows, refs, mismatched = await load_rows(args, routes, price)
    if not rows:
        raise SystemExit(f"No item has results for every route {routes} and muse_direct "
                         f"under run id {args.id!r}")
    out_of_fold(rows, routes, price)

    groups = {"all": rows, **{b: [r for r in rows if r["bench"] == b] for b in BENCHES}}
    ceilings = {g: reference(rs, refs)["muse_direct"] for g, rs in groups.items() if rs}
    sweep = []
    for lam in LAMBDAS:
        picked = choose(rows, routes, lam)
        point = {"lam": lam}
        for g, rs in groups.items():
            if rs:
                chosen = picked if g == "all" else [p for r, p in zip(rows, picked) if r["bench"] == g]
                point[g] = with_ratios(score(rs, chosen), ceilings[g])
        sweep.append(point)
    within = [p for p in sweep if p["all"]["muse_token_share"] <= args.budget]
    best = max(within or sweep[-1:], key=lambda p: (p["all"]["accuracy"], p["lam"]))  # tie: cheaper
    lam = best["lam"]

    picked = choose(rows, routes, lam)
    for bench in BENCHES:
        rs = [(r, p) for r, p in zip(rows, picked) if r["bench"] == bench]
        if rs:
            write_full_baseline(args, bench, [r for r, _ in rs], [p for _, p in rs], lam)
    model_path = RESULTS / f"gate_{args.id}_model.json"
    train(rows, routes, lam).save(model_path, trained_on=args.id, items=len(rows),
                                  budget=args.budget)

    verdicts = {}
    for bench in BENCHES:
        point = best.get(bench)
        if point:
            verdicts[bench] = {
                "accuracy_ok": point["accuracy_gap"] >= -args.margin,
                "tokens_ok": point["muse_token_share"] <= args.budget,
                "pass": point["accuracy_gap"] >= -args.margin and point["muse_token_share"] <= args.budget}
    summary = {"run_id": args.id, "routes": routes, "features": FEATURES, "items": len(rows),
               "retrieval_mismatches": mismatched, "budget": args.budget, "margin": args.margin,
               "operating_point": best, "verdicts": verdicts,
               "references": {g: {k: with_ratios(v, ceilings[g]) for k, v in reference(rs, refs).items()}
                              for g, rs in groups.items() if rs},
               "sweep": sweep}
    (RESULTS / f"gate_{args.id}.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (RESULTS / f"gate_{args.id}_sweep.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lam", "group", "accuracy", "muse_token_share", "muse_input_share",
                         "muse_usd", *(f"share_{r}" for r in routes)])
        for p in sweep:
            for g in groups:
                if g in p:
                    q = p[g]
                    writer.writerow([p["lam"], g, q["accuracy"], q["muse_token_share"],
                                     q["muse_input_share"], q["muse_usd"],
                                     *(q["routes"].get(r, 0.0) for r in routes)])

    print(f"{len(rows)} items, routes {routes}; {mismatched} retrievals differ from the run's")
    print(f"\n{'':18}{'accuracy':>10}{'tokens':>9}{'input':>8}   (shares of muse_direct's Muse tokens)")
    for g in groups:
        if g not in best:
            continue
        print(f"-- {g}")
        for name, point in [*summary["references"][g].items(), ("learned gate", best[g])]:
            print(f"{name:18}{point['accuracy']:>10.3f}{point['muse_token_share']:>9.1%}"
                  f"{point['muse_input_share'] or 0:>8.1%}")
    print(f"\nOperating point lam={lam:g}: routes {best['all']['routes']}")
    for bench, v in verdicts.items():
        p = best[bench]
        print(f"{'PASS' if v['pass'] else 'FAIL'} {bench}: accuracy {p['accuracy']:.3f} vs "
              f"muse_direct {ceilings[bench]['accuracy']:.3f} (gap {p['accuracy_gap']:+.3f}, "
              f"allowed -{args.margin}), Muse tokens {p['muse_token_share']:.1%} "
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
    asyncio.run(main(parser.parse_args()))

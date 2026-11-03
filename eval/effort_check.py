"""How much to trust the minimal-effort result: the gates at 10%, with intervals.

    uv run python -m eval.effort_check                  # full1-minimal against full1

Needs the full run's database (docker compose start) and scripts/effort_run.sh's results.
No model is called: like eval/learn_gate.py, it rebuilds the gate's rows from the saved
answers and the run's memory, so its replay is exactly learn_gate's. Writes
results/runs/effort_check_<id>.json, which results/full_run.ipynb (section 10) shows.

Added on 1 Nov, after the minimal-effort run, to check one number: on SWE-bench the
per-route gate scored 80.7% against muse_direct's 83.0% (both at minimal effort), at 9.8% of
muse_direct's Muse tokens, inside the north star's pass rule. For each benchmark:

- **Intervals:** a paired bootstrap over items (4,000 resamples, seed 0, as in the
  notebook) of the gap to muse_direct and of the token share (a ratio of sums, resampled).
- **Two references:** muse_direct at the same effort (like for like) and at default effort
  (the stronger reference, the full run's).
- **Choosing lambda without the test items:** learn_gate picks the most accurate lambda within
  10% on the same items it reports. Here each fold's lambda is picked on the other folds
  (the same team folds) and applied to it, so no item helps choose its own route.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np

from eval.common import RESULTS
from eval.learn_gate import (BENCHES, CONFIDENCE_FEATURES, FEATURES, LAMBDAS, Ledger, Route,
                             cascade_choose, cascade_out_of_fold, choose, folds, load_rows,
                             out_of_fold, run_results)
from gateway.config import Settings

B, SEED, BUDGET, MARGIN = 4000, 0, 0.10, 0.03
HEADLINE = {"swebench": "hit", "locomo": "correct"}


def tokens(r) -> int:
    return r.muse_prompt_tokens + r.muse_completion_tokens


def interval(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def compare(right: np.ndarray, spent: np.ndarray, ref_right: np.ndarray, ref_spent: np.ndarray,
            idx: np.ndarray) -> dict:
    """Gap and token share against one reference, with paired bootstrap intervals."""
    gap = right[idx].mean(axis=1) - ref_right[idx].mean(axis=1)
    share = spent[idx].sum(axis=1) / ref_spent[idx].sum(axis=1)
    return {"reference_accuracy": float(ref_right.mean()),
            "gap": float(right.mean() - ref_right.mean()), "gap_interval": interval(gap),
            "share": float(spent.sum() / ref_spent.sum()), "share_interval": interval(share),
            "p_within_margin": float((gap >= -MARGIN).mean())}


def nested(rows: list[dict], picks: dict[float, list[str]], ref_spent: np.ndarray) -> list[str]:
    """Each fold's route from the lambda that the other folds' items would have chosen."""
    fold = folds([r["team"] for r in rows])
    out = [""] * len(rows)
    for k in sorted(set(fold)):
        train = fold != k
        best = None
        for lam, picked in picks.items():
            right = np.mean([r["correct"][p] for r, p, t in zip(rows, picked, train) if t])
            spent = sum(r["prompt"][p] + r["completion"][p] for r, p, t in zip(rows, picked, train) if t)
            if spent <= BUDGET * ref_spent[train].sum() and (best is None or (right, lam) > best[:2]):
                best = (right, lam)
        lam = best[1] if best else max(picks)
        for i in np.flatnonzero(~train):
            out[i] = picks[lam][i]
    return out


async def main(args) -> None:
    settings = Settings()
    price = Ledger(None, 0.0, settings.muse_input_price_per_m, settings.muse_output_price_per_m).cost
    routes = [Route.LOCAL_MEMORY.value, Route.ESCALATE_COMPACT.value, Route.ESCALATE_RAW.value]
    run = argparse.Namespace(id=args.id, n=args.n, seed=0, database=args.database,
                             no_confidence=False)
    rows, refs, _, confident = await load_rows(run, routes, price)
    assert confident, "needs eval/confidence.py's probes"
    out_of_fold(rows, routes, price, list(FEATURES) + list(CONFIDENCE_FEATURES))
    cascade_out_of_fold(rows, list(CONFIDENCE_FEATURES))
    saved = json.loads((RESULTS / f"gate_{args.id}.json").read_text())
    rng = np.random.default_rng(SEED)

    report = {"run_id": args.id, "reference_run_id": args.reference, "resamples": B,
              "budget": BUDGET, "margin": MARGIN, "benchmarks": {}}
    for bench in BENCHES:
        rs = [r for r in rows if r["bench"] == bench]
        same = np.array([bool(refs["muse_direct"][r["item"]].score.get(HEADLINE[bench])) for r in rs])
        same_spent = np.array([tokens(refs["muse_direct"][r["item"]]) for r in rs], float)
        default = run_results(args.reference, bench, "muse_direct")
        def_right = np.array([bool(default[r["item"]].score.get(HEADLINE[bench])) for r in rs])
        def_spent = np.array([tokens(default[r["item"]]) for r in rs], float)
        idx = rng.integers(0, len(rs), size=(B, len(rs)))

        picks = {lam: choose(rs, routes, lam) for lam in LAMBDAS}
        lam = saved["operating_points"][bench]["lam"]
        threshold = saved["cascade"]["operating_points"][f"{BUDGET:g}"][bench]["threshold"]
        options = {
            "per-route gate, lambda chosen on these items": picks[lam],
            "per-route gate, lambda chosen on other folds": nested(rs, picks, same_spent),
            "cascade gate (shipped), threshold chosen on these items": cascade_choose(rs, threshold),
            "every question to Muse + summaries": [Route.ESCALATE_COMPACT.value] * len(rs),
        }
        out = {"items": len(rs), "options": {}}
        for name, picked in options.items():
            right = np.array([r["correct"][p] for r, p in zip(rs, picked)])
            spent = np.array([r["prompt"][p] + r["completion"][p] for r, p in zip(rs, picked)], float)
            out["options"][name] = {
                "accuracy": float(right.mean()),
                "kept_on_llama": picked.count(Route.LOCAL_MEMORY.value) / len(rs),
                "same_effort": compare(right, spent, same, same_spent, idx),
                "default_effort": compare(right, spent, def_right, def_spent, idx)}
        report["benchmarks"][bench] = out

    # The replay must reproduce learn_gate's saved operating point exactly.
    for bench in BENCHES:
        mine = report["benchmarks"][bench]["options"]["per-route gate, lambda chosen on these items"]
        ref = saved["operating_points"][bench]
        assert abs(mine["accuracy"] - ref["accuracy"]) < 1e-9, (bench, mine["accuracy"], ref["accuracy"])
        assert abs(mine["same_effort"]["share"] - ref["muse_token_share"]) < 1e-9, bench

    path = RESULTS / f"effort_check_{args.id}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path.relative_to(RESULTS.parent.parent)}")
    for bench, out in report["benchmarks"].items():
        print(f"\n{bench} ({out['items']} items)")
        for name, o in out["options"].items():
            s, d = o["same_effort"], o["default_effort"]
            print(f"  {name}: {o['accuracy']:.1%}"
                  f"\n    vs muse_direct, same effort {s['reference_accuracy']:.1%}: gap {100 * s['gap']:+.1f} "
                  f"[{100 * s['gap_interval'][0]:+.1f}, {100 * s['gap_interval'][1]:+.1f}], "
                  f"share {s['share']:.1%} [{s['share_interval'][0]:.1%}, {s['share_interval'][1]:.1%}], "
                  f"P(gap >= -3) {s['p_within_margin']:.0%}"
                  f"\n    vs muse_direct, default effort {d['reference_accuracy']:.1%}: gap {100 * d['gap']:+.1f} "
                  f"[{100 * d['gap_interval'][0]:+.1f}, {100 * d['gap_interval'][1]:+.1f}], "
                  f"share {d['share']:.1%} [{d['share_interval'][0]:.1%}, {d['share_interval'][1]:.1%}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", default="full1-minimal", help="the effort run's id")
    parser.add_argument("--reference", default="full1", help="the default-effort run's id")
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("--database", default="postgresql://postgres:postgres@localhost:5432/gateway_full1_full")
    asyncio.run(main(parser.parse_args()))

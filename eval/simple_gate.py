"""Can a simpler gate spot the open model's wrong drafts better than the cascade's six signals?

    uv run python -m eval.simple_gate                    # full1 (8B) and big1 (70B), code only

Needs the runs' databases (docker compose start). No model is called. Added on 3 Nov, after
the 70B run, where the self-check alone separated right drafts from wrong (AUC 0.69) better
than the cascade that combines it with five other signals (0.58).

Every gate below goes through the same pipeline as eval/learn_gate.py's cascade: a logistic
P(draft right) fitted out of fold (5 folds grouped by repository), so no issue helps score
itself, and escalating to Muse with team memory below a threshold. Code only, because the
second opinion (the other open model's draft) exists only there.

- **Signals:** the cascade's six; the self-check alone; the self-check with the number of
  files the draft names; and those plus whether the other open model names the same first
  file (a second opinion: for the 70B's drafts, the 8B's).
- **Unfitted scores:** the raw self-check, and the self-check plus 1 when the second opinion
  agrees (a fixed rule, no weights). A fitted gate's scores pool five folds whose intercepts
  differ (folds are whole repositories, with different shares of right drafts); for a single
  signal that is mostly near 0, the fold then decides the ranking more than the signal does,
  and the fitted self-check scores at chance. Ranking by the raw signal needs no fitting.
- **Thresholds:** the most accurate within each budget, chosen on these items (as the paper's
  cascade numbers are), and chosen on the other folds (nothing chooses its own route).
- **Intervals:** paired bootstrap (4,000 resamples, seed 0) of each gate's accuracy minus the
  cascade's, at the 10% budget.

Writes results/runs/simple_gate.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np

from eval.common import RESULTS
from eval.learn_gate import (CONFIDENCE_FEATURES, Ledger, Route, cascade_out_of_fold, folds,
                             load_rows, run_results)
from gateway.config import Settings

BUDGETS = (0.10, 0.20)
B, SEED = 4000, 0
RUNS = {"full1": {"label": "Llama 3.1 8B", "muse_id": None, "other": "big1"},
        "big1": {"label": "Llama 3.3 70B", "muse_id": "full1", "other": "full1"}}
LOCAL = Route.LOCAL_MEMORY.value


def files(answer: str) -> list[str]:
    return [line.strip() for line in answer.splitlines() if line.strip()]


def pick(rows: list[dict], scores: np.ndarray, budget: float, ref_spent: float) -> tuple[float, np.ndarray]:
    """Escalate the lowest-scoring drafts first: the most accurate cut within budget."""
    order = np.argsort(scores, kind="stable")
    best, up = None, np.zeros(len(rows), bool)
    spent = 0.0
    for k in range(len(rows) + 1):
        if k:
            r = rows[order[k - 1]]
            spent += r["prompt"][r["escalate_to"]] + r["completion"][r["escalate_to"]]
            up[order[k - 1]] = True
        if spent > budget * ref_spent:
            break
        right = np.array([r["correct"][r["escalate_to"] if u else LOCAL] for r, u in zip(rows, up)])
        if best is None or right.mean() > best[0]:
            best = (right.mean(), up.copy(), k)
    return best[0], best[1]


def nested(rows: list[dict], scores: np.ndarray, budget: float, spent_ref: np.ndarray) -> np.ndarray:
    """Each fold escalates as many of its own drafts as the other folds' best cut would, by share."""
    fold = folds([r["team"] for r in rows])
    up = np.zeros(len(rows), bool)
    for k in sorted(set(fold)):
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tr = [rows[i] for i in train]
        _, chosen = pick(tr, scores[train], budget, spent_ref[train].sum())
        cut = scores[train][chosen].max() if chosen.any() else -np.inf  # the other folds' threshold
        up[test] = scores[test] <= cut
    return up


def outcome(rows: list[dict], up: np.ndarray, ref_spent: float) -> dict:
    right = np.array([r["correct"][r["escalate_to"] if u else LOCAL] for r, u in zip(rows, up)])
    spent = sum(r["prompt"][r["escalate_to"]] + r["completion"][r["escalate_to"]] for r, u in zip(rows, up) if u)
    return {"accuracy": float(right.mean()), "share": float(spent / ref_spent),
            "kept": float(1 - up.mean()), "right": right}


async def main(args) -> None:
    settings = Settings()
    price = Ledger(None, 0.0, settings.muse_input_price_per_m, settings.muse_output_price_per_m).cost
    routes = [LOCAL, Route.ESCALATE_COMPACT.value, Route.ESCALATE_RAW.value]
    rng = np.random.default_rng(SEED)
    report = {"budgets": list(BUDGETS), "resamples": B, "runs": {}}
    for run_id, spec in RUNS.items():
        run = argparse.Namespace(id=run_id, n=args.n, seed=0, database=None, no_confidence=False,
                                 muse_id=spec["muse_id"])
        rows, refs, _, confident = await load_rows(run, routes, price)
        assert confident, f"{run_id} needs eval/confidence.py's probes"
        rows = [r for r in rows if r["bench"] == "swebench"]
        other = run_results(spec["other"], "swebench", "llama_only")
        rows = [r for r in rows if r["item"] in other]
        for r in rows:
            mine = files(r["draft"].answer)
            r["features"]["files_named"] = float(len(mine))
            r["features"]["second_opinion_agrees"] = float(mine[:1] == files(other[r["item"]].answer)[:1])
        spent_ref = np.array([refs["muse_direct"][r["item"]].muse_prompt_tokens
                              + refs["muse_direct"][r["item"]].muse_completion_tokens for r in rows], float)
        gates = {"cascade (6 signals, shipped)": list(CONFIDENCE_FEATURES),
                 "self-check, raw (no fitting)": lambda r: r["features"]["self_check"],
                 "self-check + 1 if second opinion agrees (no fitting)":
                     lambda r: r["features"]["self_check"] + r["features"]["second_opinion_agrees"],
                 "self-check": ["self_check"],
                 "self-check + files named": ["self_check", "files_named"],
                 "self-check + files named + second opinion": ["self_check", "files_named", "second_opinion_agrees"]}
        out = {"label": spec["label"], "items": len(rows), "draft_right": float(np.mean([r["correct"][LOCAL] for r in rows])),
               "muse_direct": float(np.mean([bool(refs["muse_direct"][r["item"]].score.get("hit")) for r in rows])),
               "gates": {}}
        base = {}
        idx = rng.integers(0, len(rows), (B, len(rows)))
        for name, features in gates.items():
            if callable(features):
                scores = np.array([features(r) for r in rows], float)
                features = "unfitted"
            else:
                cascade_out_of_fold(rows, features)
                scores = np.array([r["p_draft"] for r in rows])
            y = np.array([r["correct"][LOCAL] for r in rows])
            pos, neg = scores[y], scores[~y]
            g = {"features": features,
                 "auc": float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))}
            for budget in BUDGETS:
                _, up = pick(rows, scores, budget, spent_ref.sum())
                same = outcome(rows, up, spent_ref.sum())
                other_folds = outcome(rows, nested(rows, scores, budget, spent_ref), spent_ref.sum())
                key = f"{budget:g}"
                if name.startswith("cascade"):
                    base[key] = same["right"]
                diff = same["right"][idx].mean(axis=1) - base[key][idx].mean(axis=1)
                g[key] = {"threshold_on_these_items": {k: v for k, v in same.items() if k != "right"},
                          "threshold_on_other_folds": {k: v for k, v in other_folds.items() if k != "right"},
                          "vs_cascade": {"points": float(same["right"].mean() - base[key].mean()),
                                         "interval": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                         "p_better": float((diff > 0).mean())}}
            out["gates"][name] = g
        report["runs"][run_id] = out

    path = RESULTS / "simple_gate.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path.relative_to(RESULTS.parent.parent)}")
    for run_id, out in report["runs"].items():
        print(f"\n{out['label']} drafts ({out['items']} code issues; drafts right {out['draft_right']:.1%}; "
              f"Muse with the full context {out['muse_direct']:.1%})")
        print(f"  {'gate':54} {'AUC':>5}  " + "  ".join(f"{'at ' + format(b, '.0%'):>28}" for b in BUDGETS))
        for name, g in out["gates"].items():
            cells = []
            for b in BUDGETS:
                s, o, v = (g[f"{b:g}"][k] for k in ("threshold_on_these_items", "threshold_on_other_folds", "vs_cascade"))
                cells.append(f"{s['accuracy']:.1%} ({o['accuracy']:.1%} nested) {100 * v['points']:+.1f}")
            print(f"  {name:54} {g['auc']:5.3f}  " + "  ".join(f"{c:>28}" for c in cells))
        for name, g in out["gates"].items():
            if name.startswith("cascade"):
                continue
            v = g["0.1"]["vs_cascade"]
            print(f"  vs cascade at 10%, {name}: {100 * v['points']:+.1f} [{100 * v['interval'][0]:+.1f}, "
                  f"{100 * v['interval'][1]:+.1f}], better in {v['p_better']:.0%} of resamples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=300)
    asyncio.run(main(parser.parse_args()))

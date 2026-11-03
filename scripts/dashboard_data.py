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
- Open models: the same code questions with Llama 3.3 70B drafting instead of 8B (run big1,
  added after the run; same memory, same Muse answers), with paired bootstrap intervals.
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


def raw(run_id: str, baseline: str) -> dict[str, dict]:
    path = RESULTS / "raw" / f"swebench_{baseline}_{run_id}-swebench-{baseline}.jsonl"
    return {r["item_id"]: r for r in map(json.loads, path.open())}


def hits(run_id: str, baseline: str) -> dict[str, bool]:
    return {i: bool(r["score"]["hit"]) for i, r in raw(run_id, baseline).items()}


def perfect_check(base: str, run_id: str) -> dict:
    """Escalate exactly the drafts that are wrong (to Muse with summaries): the best any gate
    that judges only the draft could do, and what it would cost."""
    draft, compact, direct = hits(run_id, "llama_only"), raw(base, "escalate_compact"), raw(base, "muse_direct")
    tokens = lambda r: r["muse_prompt_tokens"] + r["muse_completion_tokens"]  # noqa: E731
    wrong = [i for i in draft if not draft[i]]
    return {"wrong_drafts": len(wrong) / len(draft),
            "accuracy": sum(draft[i] or bool(compact[i]["score"]["hit"]) for i in draft) / len(draft),
            "share": round(sum(tokens(compact[i]) for i in wrong) / sum(tokens(direct[i]) for i in draft), 5)}


def open_models(base: str, big: str, resamples: int = 4000, seed: int = 0) -> dict | None:
    """Code (SWE-bench) with each open model drafting: alone, and in the cascade at 10% and 20%
    of Muse's tokens. Paired bootstrap of the 70B's gain over the 8B on the same items."""
    if not (RESULTS / f"gate_{big}.json").exists():
        return None
    direct = hits(base, "muse_direct")
    rng = np.random.default_rng(seed)
    out = {"muse_direct": sum(direct.values()) / len(direct), "models": [], "gain": {}}
    for run_id in (base, big):
        gate = json.loads((RESULTS / f"gate_{run_id}.json").read_text())
        model = json.loads((RESULTS / f"swebench_llama_only_{run_id}-swebench-llama_only.json").read_text())
        points = gate["cascade"]["operating_points"]
        out["models"].append({
            "run_id": run_id, "model": model["llama_model"],
            "alone": gate["references"]["swebench"]["llama_only"]["accuracy"],
            "cascade": {b: {"accuracy": points[b]["swebench"]["accuracy"],
                            "share": round(points[b]["swebench"]["muse_token_share"], 5),
                            "kept": points[b]["swebench"]["routes"].get("local_memory", 0.0)}
                        for b in ("0.1", "0.2")},
            "perfect_check": perfect_check(base, run_id)})
    for name, baseline in (("alone", "llama_only"), ("cascade_10", "full")):
        a, b = hits(base, baseline), hits(big, baseline)
        ids = sorted(set(a) & set(b))
        x, y = (np.array([d[i] for i in ids], float) for d in (a, b))
        idx = rng.integers(0, len(ids), (resamples, len(ids)))
        diff = y[idx].mean(axis=1) - x[idx].mean(axis=1)
        out["gain"][name] = {"points": float(y.mean() - x.mean()),
                             "interval": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                             "p_better": float((diff > 0).mean())}
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
            "handcheck": handcheck(args.id), "open_models": open_models(args.id, args.big)}
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
    parser.add_argument("--big", default="big1", help="the run with a larger open model drafting")
    main(parser.parse_args())

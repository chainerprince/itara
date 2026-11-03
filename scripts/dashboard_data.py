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
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
            "handcheck": handcheck(args.id)}
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

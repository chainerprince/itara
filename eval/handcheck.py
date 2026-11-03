"""The hand spot-check of 50 LoCoMo judge verdicts (the plan's check on the LLM judge).

    uv run python -m eval.handcheck make --id full1      # writes the sheet
    # fill in the `human` column: CORRECT or WRONG (leave blank to skip a row)
    uv run python -m eval.handcheck score --id full1     # agreement with the judge

The sheet samples 50 answers at random (seeded) across the LoCoMo conditions of a full run,
so every condition's verdicts get checked, and hides nothing: the judge's verdict is shown
next to the blank column. Judge the response against the gold answer as the judge was told
to: same answer, ignoring wording and extra detail; dates must refer to the same time.
Writes and reads results/runs/handcheck_locomo_<id>.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import random

from eval import locomo
from eval.common import RESULTS

CONDITIONS = ("muse_direct", "escalate_raw", "escalate_compact", "llama_only")
FIELDS = ("n", "item_id", "condition", "question", "gold", "response", "judge", "human", "note")


def sheet_path(run_id: str):
    return RESULTS / f"handcheck_locomo_{run_id}.csv"


def make(args) -> None:
    items = {i.id: i for i in locomo.load(args.n, args.seed)}
    answers = []
    for c in CONDITIONS:
        path = RESULTS / "raw" / f"locomo_{c}_{args.id}-locomo-{c}.jsonl"
        if path.exists():
            answers += [(c, json.loads(line)) for line in path.read_text().splitlines()]
    picked = random.Random(args.seed).sample(answers, min(args.k, len(answers)))
    path = sheet_path(args.id)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists (it may hold your verdicts): pass --force to replace it")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for n, (c, r) in enumerate(picked, 1):
            item = items[r["item_id"]]
            writer.writerow({"n": n, "item_id": r["item_id"], "condition": c,
                             "question": item.meta["question"], "gold": item.gold,
                             "response": r["answer"], "judge": r["score"]["judge"],
                             "human": "", "note": ""})
    print(f"Wrote {path}: {len(picked)} verdicts to check")


def score(args) -> None:
    rows = list(csv.DictReader(sheet_path(args.id).open()))
    done = [r for r in rows if r["human"].strip()]
    if not done:
        raise SystemExit("No row has a `human` verdict yet")
    judge = [locomo.parse_verdict(r["judge"]) for r in done]
    human = [locomo.parse_verdict(r["human"]) for r in done]
    agree = sum(j == h for j, h in zip(judge, human))
    print(f"{len(done)}/{len(rows)} checked; the judge agrees on {agree} ({agree / len(done):.0%})")
    print(f"  judge CORRECT, human WRONG (judge too lenient): "
          f"{sum(j and not h for j, h in zip(judge, human))}")
    print(f"  judge WRONG, human CORRECT (judge too strict):  "
          f"{sum(h and not j for j, h in zip(judge, human))}")
    for c in CONDITIONS:
        sub = [(j, h) for r, j, h in zip(done, judge, human) if r["condition"] == c]
        if sub:
            print(f"  {c:17} {sum(j == h for j, h in sub)}/{len(sub)} agree")
    for r, j, h in zip(done, judge, human):
        if j != h:
            print(f"  #{r['n']} {r['condition']}: judge {r['judge']}, human {r['human']}"
                  f" ({r['note'] or 'no note'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["make", "score"])
    parser.add_argument("--id", required=True, help="the FULL_ID of scripts/full_run.sh")
    parser.add_argument("-n", type=int, default=300, help="the run's LoCoMo items")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-k", type=int, default=50, help="verdicts to check")
    parser.add_argument("--force", action="store_true", help="replace an existing sheet")
    args = parser.parse_args()
    make(args) if args.action == "make" else score(args)

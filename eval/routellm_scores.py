"""An off-the-shelf router's scores for every question, to compare with Itara's cascade.

RouteLLM's BERT router (Ong et al., 2025; checkpoint routellm/bert_gpt4_augmented) reads only
the question and gives the probability that the strong model is needed. It is trained on
Chatbot Arena preferences, not on these benchmarks, and is used as shipped. RouteLLM's
dependencies conflict with the project's, so scoring runs in its own environment:

    uv run python -m eval.routellm_scores export                 # questions -> scratch file
    OPENAI_API_KEY=unused uvx --python 3.11 --with routellm --with "pyarrow<20" \
        python eval/routellm_scores.py score                     # the key is never used
    uv run python -m eval.cost_compare                           # prices it beside the cascade

Writes results/runs/routellm_bert_<id>.jsonl: {"item", "bench", "strong_win_rate"} per line.
Added on 3 Nov, after the run. No model is called but the router itself (on the CPU).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "results" / "runs"
QUESTIONS = RUNS / "routellm_questions.jsonl"  # not committed: it is the benchmarks' text
CHECKPOINT = "routellm/bert_gpt4_augmented"


def export(n: int = 300, seed: int = 0) -> None:
    from eval import locomo, swebench_loc

    with QUESTIONS.open("w") as f:
        for bench, items in (("swebench", swebench_loc.load(n, seed)), ("locomo", locomo.load(n, seed))):
            for item in items:
                f.write(json.dumps({"item": item.id, "bench": bench, "question": item.question}) + "\n")
    print(f"wrote {QUESTIONS}")


def score(run_id: str = "full1") -> None:
    from routellm.routers.routers import BERTRouter

    router = BERTRouter(checkpoint_path=CHECKPOINT)
    out = RUNS / f"routellm_bert_{run_id}.jsonl"
    with QUESTIONS.open() as src, out.open("w") as f:
        for line in src:
            q = json.loads(line)
            win = float(router.calculate_strong_win_rate(q["question"]))
            f.write(json.dumps({"item": q["item"], "bench": q["bench"], "strong_win_rate": win}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()

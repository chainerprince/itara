"""SWE-bench Lite issue localisation: given the issue, name the files the real patch changes.

Uses princeton-nlp/SWE-bench_Lite_bm25_13K, which already holds each issue with ~13K tokens
of BM25-retrieved files, so no repositories are cloned. Each retrieved file becomes a team
memory (the team is the repo). The memory source is the path plus a hash of the contents, so
tasks on other commits reuse unchanged files and changed files are stored as new memories.
No LLM judge: the gold answer is the set of files in the patch.
"""

from __future__ import annotations

import hashlib
import random
import re

from eval.common import Item

DATASET = "princeton-nlp/SWE-bench_Lite_bm25_13K"
MAX_FILES = 5
INSTRUCTION = (
    f"List the files in this repository that must be changed to fix the issue above, one path "
    f"per line, most likely first, at most {MAX_FILES}. Reply with the paths only."
)

_FILE = re.compile(r"\[start of (.+?)\]\n(.*?)\n\[end of \1\]", re.DOTALL)
_PATH = re.compile(r"[\w.\-]+(?:/[\w.\-]+)*\.\w+")


def gold_files(patch: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"^diff --git a/(\S+) b/", patch, re.MULTILINE)))


def retrieved_files(text: str) -> list[tuple[str, str]]:
    return _FILE.findall(text)


def predicted_files(answer: str) -> list[str]:
    """Paths from a free-form answer, in order, without duplicates."""
    paths = (p.removeprefix("./") for p in _PATH.findall(answer.replace("`", " ")))
    return list(dict.fromkeys(paths))[:MAX_FILES]


def score(item: Item, answer: str) -> dict:
    gold, predicted = set(item.gold), set(predicted_files(answer))
    return {
        "hit": gold <= predicted,
        "recall": len(gold & predicted) / len(gold),
        "ceiling": gold <= set(item.meta["retrieved"]),
    }


def to_item(row: dict) -> Item:
    text = row["text"]
    files = retrieved_files(text)
    end = text.find("</code>")
    context = text[: end + len("</code>")] if end >= 0 else text
    # The team is the repo, so the source is just the path (+ content hash): the model sees
    # exactly the path it should answer with.
    docs = [(f"{path}@{hashlib.sha256(body.encode()).hexdigest()[:8]}",
             f"File {path}:\n{body}") for path, body in files]
    return Item(
        id=row["instance_id"],
        team_id=f"eval:{row['repo']}",
        docs=docs,
        question=f"<issue>\n{row['problem_statement']}\n</issue>\n\n{INSTRUCTION}",
        raw_context=f"{context}\n\n{INSTRUCTION}",
        gold=gold_files(row["patch"]),
        meta={"retrieved": [path for path, _ in files]},
    )


def load(n: int, seed: int = 0, repo: str | None = None) -> list[Item]:
    from datasets import load_dataset

    rows = [r for r in load_dataset(DATASET, split="test") if repo is None or r["repo"] == repo]
    rows.sort(key=lambda r: r["instance_id"])
    random.Random(seed).shuffle(rows)
    return [to_item(r) for r in rows[:n]]

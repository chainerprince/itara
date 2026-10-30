"""LoCoMo subset: questions about long two-person conversations, graded by an LLM judge.

Each conversation is one team and each session one memory, labelled with the session's
date: the date is said once at the start of a session, and a summary of a later page (or a
summary that drops it) would otherwise leave "last Friday" unanswerable. Categories 1-4 only; category 5
(adversarial) has no gold answer. The judge is the "Muse" model, called through the spend
cap and recorded as route "judge", separate from the answering tokens. Verdicts are kept
in the JSONL for the hand spot-check of 50.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import httpx

from eval.common import Harness, Item
from memory.store import Scope

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA = Path(__file__).parent / "data" / "locomo10.json"
INSTRUCTION = "Answer the question in a short phrase, using what was said in the conversation."
# Muse Spark is a reasoning model and max_tokens counts its reasoning: a verdict takes ~70
# tokens at minimal effort (~250 at the default), so leave room or the verdict comes back empty.
JUDGE_MAX_TOKENS = 1024
JUDGE_PROMPT = """\
Question: {question}
Gold answer: {gold}
Response: {answer}

Does the response give the same answer as the gold answer? Ignore wording and extra detail;
dates must refer to the same time. Reply with exactly one word: CORRECT or WRONG."""


def turns_by_id(conversation: dict) -> dict[str, str]:
    """dia_id -> the turn exactly as it appears in a session doc."""
    return {t["dia_id"]: f"{t['speaker']}: {t['text']}"
            for key, value in conversation.items()
            if key.startswith("session_") and isinstance(value, list)
            for t in value if "dia_id" in t}


def session_of_turn(conversation: dict) -> dict[str, str]:
    """dia_id -> the name of the session it was said in."""
    return {t["dia_id"]: key
            for key, value in conversation.items()
            if key.startswith("session_") and isinstance(value, list)
            for t in value if "dia_id" in t}


def sessions(conversation: dict) -> list[tuple[str, str, str]]:
    """(name, date, text) for each session in order; the text has the date and labelled turns."""
    out = []
    n = 1
    while f"session_{n}" in conversation:
        turns = "\n".join(f"{t['speaker']}: {t['text']}" for t in conversation[f"session_{n}"])
        date = conversation.get(f"session_{n}_date_time", "")
        out.append((f"session_{n}", date, f"Conversation on {date}\n{turns}"))
        n += 1
    return out


def items_from(samples: list[dict]) -> list[Item]:
    items = []
    for sample in samples:
        labels = {name: f"locomo:{sample['sample_id']}:{name}" + (f", {date}" if date else "")
                  for name, date, _ in sessions(sample["conversation"])}
        docs = [(labels[name], text) for name, _, text in sessions(sample["conversation"])]
        transcript = "\n\n".join(text for _, text in docs)
        turns = turns_by_id(sample["conversation"])
        session = session_of_turn(sample["conversation"])
        for i, qa in enumerate(sample["qa"]):
            if qa["category"] == 5 or "answer" not in qa:
                continue
            question = f"{qa['question']}\n\n{INSTRUCTION}"
            items.append(Item(
                id=f"{sample['sample_id']}:{i}",
                team_id=f"eval:locomo:{sample['sample_id']}",
                docs=docs,
                question=question,
                raw_context=f"{transcript}\n\n{question}",
                gold=str(qa["answer"]),
                meta={"question": qa["question"], "category": qa["category"]},
                # Exact turns only: a paraphrased memory summary counts as not seen.
                evidence=[[turns[d]] for d in qa.get("evidence", []) if d in turns],
                # So retrieval is still measurable: memory from the evidence's sessions.
                evidence_sources=sorted({labels[session[d]]
                                         for d in qa.get("evidence", []) if d in session}),
            ))
    return items


def load(n: int, seed: int = 0) -> list[Item]:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_bytes(httpx.get(URL, follow_redirects=True, timeout=60).raise_for_status().content)
    items = items_from(json.loads(DATA.read_text()))
    picked = random.Random(seed).sample(items, min(n, len(items)))
    return sorted(picked, key=lambda item: item.team_id)  # one conversation's items together


def parse_verdict(text: str) -> bool:
    return text.strip().upper().startswith("CORRECT")


async def judge(harness: Harness, item: Item, answer: str, user: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=item.meta["question"], gold=item.gold, answer=answer)
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": JUDGE_MAX_TOKENS,
            "temperature": 0}
    if not harness.services.muse_is_standin:  # the local stand-in isn't a reasoning model
        body["reasoning_effort"] = "minimal"
    response, _ = await harness.paid_call(Scope(item.team_id, user), "judge", body)
    verdict = response["choices"][0]["message"].get("content") or ""
    return {"correct": parse_verdict(verdict), "judge": verdict.strip()}

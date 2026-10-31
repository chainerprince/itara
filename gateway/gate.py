"""Decide, per turn, whether to answer locally or escalate to the closed model.

`rule_gate` is the rule-based baseline: it decides before any model runs. `LearnedGate` is
a cascade: Llama answers with memory first (cheap, and cached when memory is shared), then
the gate looks at the question, the retrieved memory and that draft, and either returns the
draft or escalates. For each route it predicts the chance of a right answer (one logistic
model per route, trained by eval/learn_gate.py on escalation regret) and picks the route
with the best chance minus `lam` times its estimated Muse cost in dollars. Sweeping `lam`
draws the cost/accuracy curve; the saved gate keeps one point on it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from memory.store import Memory, estimate_tokens


class Route(StrEnum):
    LOCAL = "local"                        # Llama, no memory
    LOCAL_MEMORY = "local_memory"          # Llama + memory summaries
    ESCALATE_COMPACT = "escalate_compact"  # Muse + memory summaries
    ESCALATE_RAW = "escalate_raw"          # Muse + raw pages behind the summaries
    PASSTHROUGH = "passthrough"            # tool calls: sent to Muse unchanged


ESCALATIONS = {Route.ESCALATE_COMPACT, Route.ESCALATE_RAW, Route.PASSTHROUGH}


def has_tools(body: dict) -> bool:
    if body.get("tools") or body.get("functions"):
        return True
    return any(m.get("role") == "tool" or m.get("tool_calls") for m in body.get("messages", []))


def rule_gate(question: str, hits: list[Memory]) -> Route:
    """Escalate long or code-heavy questions; otherwise answer locally."""
    if estimate_tokens(question) > 200 or "```" in question:
        return Route.ESCALATE_COMPACT
    return Route.LOCAL_MEMORY if hits else Route.LOCAL


# What the gate sees: the question, the memory retrieved for it, and Llama's draft answer.
FEATURES = (
    "question_tokens",   # log of the question's length
    "code_question",     # code, a traceback or a file path in the question
    "hits",              # memories retrieved, out of 5
    "distinct_sources",  # share of retrieved memories from different docs
    "top_similarity",    # best cosine between the question and a memory
    "mean_similarity",
    "draft_unsure",      # the draft says the answer isn't there
    "draft_tokens",      # log of the draft's length
    "draft_grounding",   # share of the draft's words found in the memory
    "draft_similarity",  # best cosine between the draft and a memory
    "number_missing",    # asks when / how many, and the draft has no number
)
UNSURE = re.compile(r"\b(no (information|mention|record|details?)|not (mentioned|specified|provided|"
                    r"stated|given|clear)|does(n't| not) (say|mention|state|specify)|"
                    r"(don't|do not|cannot|can't) (know|determine|tell)|unknown|unclear)\b", re.I)
CODE = re.compile(r"```|Traceback|\bdef |\bclass |\w+/\w+\.\w+")
NUMBER_QUESTION = re.compile(r"^\s*(when|how (many|much|long|old)|what (year|date|time))\b", re.I)
WORD = re.compile(r"[a-z0-9_./]{4,}")
COMPACT_OVERHEAD_TOKENS = 40  # the memory header and source labels


def _cosines(vec: np.ndarray, others: list[np.ndarray]) -> list[float]:
    return [float(vec @ o / (np.linalg.norm(vec) * np.linalg.norm(o) or 1)) for o in others]


def gate_features(question: str, hits: list[Memory], draft: str, vectors: list[np.ndarray],
                  ) -> dict[str, float]:
    """`vectors`: embeddings of the question, the draft, then each hit's content, in order."""
    q_vec, d_vec, hit_vecs = vectors[0], vectors[1], vectors[2:]
    q_sims, d_sims = _cosines(q_vec, hit_vecs), _cosines(d_vec, hit_vecs)
    memory_text = " ".join(f"{h.source or ''} {h.content}" for h in hits).lower()
    draft_words = set(WORD.findall(draft.lower()))
    return {
        "question_tokens": math.log1p(estimate_tokens(question)),
        "code_question": float(bool(CODE.search(question))),
        "hits": len(hits) / 5,
        "distinct_sources": len({h.source for h in hits}) / len(hits) if hits else 0.0,
        "top_similarity": max(q_sims, default=0.0),
        "mean_similarity": sum(q_sims) / len(q_sims) if q_sims else 0.0,
        "draft_unsure": float(bool(UNSURE.search(draft))),
        "draft_tokens": math.log1p(estimate_tokens(draft)),
        "draft_grounding": (sum(w in memory_text for w in draft_words) / len(draft_words)
                            if draft_words else 0.0),
        "draft_similarity": max(d_sims, default=0.0),
        "number_missing": float(bool(NUMBER_QUESTION.search(question))
                                and not re.search(r"\d", draft)),
    }


def estimated_prompt_tokens(question: str, hits: list[Memory], pages: list[Memory],
                            ) -> dict[str, int]:
    """Muse prompt tokens each route would send (before it's called)."""
    q = estimate_tokens(question)
    return {Route.LOCAL_MEMORY: 0,
            Route.ESCALATE_COMPACT: q + COMPACT_OVERHEAD_TOKENS + sum(h.token_count for h in hits),
            Route.ESCALATE_RAW: q + COMPACT_OVERHEAD_TOKENS + sum(p.token_count for p in pages)}


@dataclass
class LearnedGate:
    routes: list[str]                 # the routes it chooses between; the first is local
    mean: list[float]                 # feature standardisation
    scale: list[float]
    weights: dict[str, list[float]]   # route -> coefficients per feature, then the intercept
    output_tokens: float              # Muse's mean completion tokens per escalation
    lam: float                        # accuracy traded per dollar of Muse

    @classmethod
    def load(cls, path: str | Path) -> LearnedGate:
        data = json.loads(Path(path).read_text())
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})

    def save(self, path: str | Path, **extra) -> None:
        Path(path).write_text(json.dumps({**self.__dict__, "features": FEATURES, **extra},
                                         indent=2) + "\n")

    def probabilities(self, features: dict[str, float]) -> dict[str, float]:
        x = (np.array([features[f] for f in FEATURES]) - self.mean) / self.scale
        return {r: float(1 / (1 + np.exp(-(x @ np.array(w[:-1]) + w[-1]))))
                for r, w in self.weights.items()}

    def costs(self, prompt_tokens: dict[str, int], price: Callable[[int, float], float],
              ) -> dict[str, float]:
        """Estimated Muse dollars per route; `price` is Ledger.cost(prompt, completion)."""
        return {r: 0.0 if i == 0 else price(prompt_tokens[r], self.output_tokens)
                for i, r in enumerate(self.routes)}

    def decide(self, probabilities: dict[str, float], costs: dict[str, float],
               lam: float | None = None) -> Route:
        lam = self.lam if lam is None else lam
        return Route(max(self.routes, key=lambda r: probabilities[r] - lam * costs[r]))

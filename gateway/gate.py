"""Decide, per turn, whether to answer locally or escalate to the closed model.

`rule_gate` is the rule-based baseline: it decides before any model runs. `LearnedGate` is
a cascade: Llama answers with memory first (cheap, and cached when memory is shared), then
the gate looks at the question, the retrieved memory and that draft, and either returns the
draft or escalates. For each route it predicts the chance of a right answer (one logistic
model per route, trained by eval/learn_gate.py on escalation regret) and picks the route
with the best chance minus `lam` times its estimated Muse cost in dollars. Sweeping `lam`
draws the cost/accuracy curve; the saved gate keeps one point on it.

Features of the question, memory and draft alone barely beat escalating at random (the
first full run), so the gate can also ask Llama how sure it is (`draft_confidence`): the
probabilities it gave the draft's tokens, whether resampled answers agree with the draft,
and its P(yes) when asked whether the draft is right. That costs two more Llama calls on
the draft's cached prefix, and no Muse.

`CascadeGate` is the simpler gate the full run favoured: one logistic model scores P(draft
right) from Llama's confidence alone, and a draft under `threshold` escalates. Code memory
escalates with its summaries (Muse knows the code; summaries matched full context), other
memory with the raw pages (summaries of conversations and docs drop the details asked
about). On the full run it beat the per-route LearnedGate by 6 points at 10% of the tokens
on code (results/full_run.ipynb, section 8).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from memory.store import Memory, estimate_tokens


class Route(StrEnum):
    LOCAL = "local"                        # Llama, no memory
    LOCAL_MEMORY = "local_memory"          # Llama + memory summaries
    ESCALATE_COMPACT = "escalate_compact"  # Muse + memory summaries
    ESCALATE_RAW = "escalate_raw"          # Muse + raw pages behind the summaries
    ESCALATE_PAGES = "escalate_pages"      # Muse + raw pages searched directly, no summaries
    PASSTHROUGH = "passthrough"            # tool calls with no team memory: Muse, unchanged


ESCALATIONS = {Route.ESCALATE_COMPACT, Route.ESCALATE_RAW, Route.ESCALATE_PAGES,
               Route.PASSTHROUGH}


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
    "draft_names_source",  # share of file paths in the draft that are retrieved sources
)
# Llama's own confidence in its draft (draft_confidence): needs logprobs and two more calls.
CONFIDENCE_FEATURES = (
    "draft_logprob_mean",    # mean log-probability of the draft's tokens
    "draft_logprob_min",     # its least likely token
    "draft_unlikely_share",  # share of the draft's tokens Llama gave under 50%
    "self_check",            # P(yes) when asked whether the draft is right
    "sample_agreement",      # mean cosine between resampled answers and the draft
    "sample_same_answer",    # share of resampled answers with the draft's first line
)
SAMPLES = 4
SAMPLE_TEMPERATURE = 1.0
# A runaway draft (Llama repeating itself to the context limit) mustn't push the probes past
# it: resamples stop at SAMPLE_MAX_TOKENS, and the self-check sees the draft's start only.
SAMPLE_MAX_TOKENS = 512
CHECK_DRAFT_CHARS = 4000
# An open model that reasons (Muse Glimmer) writes a reasoning channel before its answer. vLLM's
# reasoning parser moves it to message.reasoning, but the logprobs still cover both, so the
# features read the answer's tokens only, and the probes get room to reason first.
REASONING_MAX_TOKENS = 4096
SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")
SELF_CHECK = ("Is your answer above correct and supported by the team memory? "
              "Reply with only yes or no.")
UNSURE = re.compile(r"\b(no (information|mention|record|details?)|not (mentioned|specified|provided|"
                    r"stated|given|clear)|does(n't| not) (say|mention|state|specify)|"
                    r"(don't|do not|cannot|can't) (know|determine|tell)|unknown|unclear)\b", re.I)
CODE = re.compile(r"```|Traceback|\bdef |\bclass |\w+/\w+\.\w+")
NUMBER_QUESTION = re.compile(r"^\s*(when|how (many|much|long|old)|what (year|date|time))\b", re.I)
WORD = re.compile(r"[a-z0-9_./]{4,}")
PATH = re.compile(r"[\w.-]+(?:/[\w.-]+)+\.\w+")
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
    paths = set(PATH.findall(draft))
    sources = {(h.source or "").split("@")[0] for h in hits}  # code sources end in @<commit>
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
        "draft_names_source": (sum(p in sources for p in paths) / len(paths) if paths else 0.0),
    }


def reasoning_of(response: dict) -> str:
    """The reasoning before the answer, for a model that reasons; "" otherwise."""
    message = (response.get("choices") or [{}])[0].get("message") or {}
    return message.get("reasoning") or message.get("reasoning_content") or ""


def answer_logprobs(response: dict) -> list[dict]:
    """The answer's logprob entries: all of them, or after reasoning, the last ones, which
    spell the message's content."""
    choice = (response.get("choices") or [{}])[0]
    entries = (choice.get("logprobs") or {}).get("content") or []
    if not reasoning_of(response):
        return entries
    length = len(((choice.get("message") or {}).get("content") or "").strip())
    if not length:
        return []
    entries = [t for t in entries if SPECIAL_TOKEN.sub("", t["token"])]
    text = ""
    for i in range(len(entries) - 1, -1, -1):
        text = SPECIAL_TOKEN.sub("", entries[i]["token"]) + text
        if len(text.strip()) >= length:
            return entries[i:]
    return entries


def token_logprobs(response: dict) -> list[float]:
    """The chosen tokens' log-probabilities in the answer, if the response carries them."""
    return [t["logprob"] for t in answer_logprobs(response) if t.get("logprob") is not None]


def yes_probability(response: dict) -> float:
    """P(yes) / (P(yes) + P(no)) from the answer's first token's top logprobs (after reasoning,
    its first yes or no); 0.5 if neither shows."""
    content = answer_logprobs(response)
    if not content:
        return 0.5
    if reasoning_of(response):
        content = [next((t for t in content if t["token"].strip().lower() in ("yes", "no")),
                        content[0])]
    top = content[0].get("top_logprobs") or [content[0]]
    p = {"yes": 0.0, "no": 0.0}
    for t in top:
        word = t["token"].strip().lower()
        if word in p:
            p[word] += math.exp(t["logprob"])
    return p["yes"] / (p["yes"] + p["no"]) if p["yes"] + p["no"] else 0.5


def first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return re.sub(r"\W+", " ", lines[0].lower()).strip() if lines else ""


async def draft_confidence(chat: Callable[[dict], Awaitable[dict]],
                           embed: Callable[[list[str]], Awaitable[list[np.ndarray]]],
                           body: dict, draft_response: dict) -> dict[str, float]:
    """Llama's confidence in its draft. `body`: the request the draft answered, memory and
    cache scope included; `draft_response`: its answer, requested with logprobs. Resamples and
    a yes/no self-check run on the same prefix, so the KV cache serves most of their prompt."""
    draft = draft_response["choices"][0]["message"].get("content") or ""
    base = {k: v for k, v in body.items() if k not in ("logprobs", "top_logprobs", "n")}
    if reasoning_of(draft_response):
        written = (draft_response.get("usage") or {}).get("completion_tokens", 0)
        sample_max = min(REASONING_MAX_TOKENS, max(SAMPLE_MAX_TOKENS, 2 * written))
        check_max = REASONING_MAX_TOKENS
    else:
        sample_max = min(SAMPLE_MAX_TOKENS, max(64, 2 * estimate_tokens(draft)))
        check_max = 1
    samples, check = await asyncio.gather(
        chat({**base, "n": SAMPLES, "temperature": SAMPLE_TEMPERATURE, "max_tokens": sample_max}),
        chat({**base, "temperature": 0, "max_tokens": check_max, "logprobs": True, "top_logprobs": 10,
              "messages": [*base["messages"],
                           {"role": "assistant", "content": draft[:CHECK_DRAFT_CHARS]},
                           {"role": "user", "content": SELF_CHECK}]}))
    texts = [c["message"].get("content") or "" for c in samples.get("choices", [])]
    vectors = await embed([draft, *texts]) if texts else []
    logprobs = token_logprobs(draft_response)
    return {
        "draft_logprob_mean": sum(logprobs) / len(logprobs) if logprobs else 0.0,
        "draft_logprob_min": min(logprobs, default=0.0),
        "draft_unlikely_share": (sum(lp < math.log(0.5) for lp in logprobs) / len(logprobs)
                                 if logprobs else 0.0),
        "self_check": yes_probability(check),
        "sample_agreement": (sum(_cosines(vectors[0], vectors[1:])) / len(texts) if texts else 1.0),
        "sample_same_answer": (sum(first_line(t) == first_line(draft) for t in texts) / len(texts)
                               if texts else 1.0),
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
    features: list[str] = field(default_factory=lambda: list(FEATURES))  # in weight order

    @classmethod
    def load(cls, path: str | Path) -> LearnedGate:
        data = json.loads(Path(path).read_text())
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def save(self, path: str | Path, **extra) -> None:
        Path(path).write_text(json.dumps({**self.__dict__, **extra}, indent=2) + "\n")

    @property
    def needs_confidence(self) -> bool:
        return any(f in CONFIDENCE_FEATURES for f in self.features)

    def probabilities(self, features: dict[str, float]) -> dict[str, float]:
        x = (np.array([features[f] for f in self.features]) - self.mean) / self.scale
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


def is_code(memory: Memory) -> bool:
    return "@" in (memory.source or "")  # code sources end in @<version>


def escalation_route(hits: list[Memory]) -> Route:
    """Summaries for code memory, raw pages for anything else (conversations, docs)."""
    code = sum(map(is_code, hits))
    return Route.ESCALATE_COMPACT if hits and code * 2 >= len(hits) else Route.ESCALATE_RAW


@dataclass
class CascadeGate:
    mean: list[float]       # feature standardisation
    scale: list[float]
    weights: list[float]    # P(draft right): coefficients per feature, then the intercept
    threshold: float        # a draft less likely right than this escalates
    features: list[str] = field(default_factory=lambda: list(CONFIDENCE_FEATURES))
    kind: str = "cascade"

    @classmethod
    def load(cls, path: str | Path) -> CascadeGate:
        data = json.loads(Path(path).read_text())
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def save(self, path: str | Path, **extra) -> None:
        Path(path).write_text(json.dumps({**self.__dict__, **extra}, indent=2) + "\n")

    @property
    def needs_confidence(self) -> bool:
        return any(f in CONFIDENCE_FEATURES for f in self.features)

    def p_right(self, features: dict[str, float]) -> float:
        x = (np.array([features[f] for f in self.features]) - self.mean) / self.scale
        return float(1 / (1 + np.exp(-(x @ np.array(self.weights[:-1]) + self.weights[-1]))))

    def decide(self, p_right: float, hits: list[Memory], threshold: float | None = None) -> Route:
        threshold = self.threshold if threshold is None else threshold
        return Route.LOCAL_MEMORY if p_right >= threshold else escalation_route(hits)


def load_gate(path: str | Path) -> LearnedGate | CascadeGate:
    """Whichever gate eval/learn_gate.py saved there."""
    kind = json.loads(Path(path).read_text()).get("kind")
    return (CascadeGate if kind == "cascade" else LearnedGate).load(path)

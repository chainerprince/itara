import math
from uuid import uuid4

import numpy as np
import pytest

from eval.learn_gate import (auc, cascade_choose, cascade_point, fit_logistic, folds,
                              operating_point, train_cascade)
from gateway.accounting import Ledger
from gateway.gate import (CHECK_DRAFT_CHARS, CONFIDENCE_FEATURES, FEATURES, REASONING_MAX_TOKENS,
                          SAMPLE_MAX_TOKENS, SAMPLES, CascadeGate, LearnedGate, Route, draft_confidence,
                          escalation_route, estimated_prompt_tokens, gate_features, load_gate,
                          token_logprobs, yes_probability)
from memory.store import Memory


def memory(content, source="notes", tokens=10):
    return Memory(id=uuid4(), owner_id=None, kind="summary", parent_id=None, source=source,
                  content=content, keywords=[], token_count=tokens)


def constant_gate(p_local, p_escalate, lam=0.0, features=FEATURES):
    """A gate whose P(right) per route ignores the features."""
    logit = lambda p: math.log(p / (1 - p))  # noqa: E731
    zeros = [0.0] * len(features)
    return LearnedGate(routes=["local_memory", "escalate_compact"], mean=zeros, scale=[1.0] * len(features),
                       weights={"local_memory": [*zeros, logit(p_local)],
                                "escalate_compact": [*zeros, logit(p_escalate)]},
                       output_tokens=1000, lam=lam, features=list(features))


def response(text, logprobs=None, choices=1):
    choice = {"message": {"role": "assistant", "content": text}}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"choices": [choice] * choices}


def test_features_of_an_unsure_and_a_grounded_draft():
    hits = [memory("Caroline went to a council meeting on 14 July 2023", "s1"), memory("pets", "s1")]
    vectors = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    question = "When did Caroline go to the meeting?"
    unsure = gate_features(question, hits, "The conversation does not mention it.", vectors)
    assert set(unsure) == set(FEATURES)
    assert unsure["draft_unsure"] == 1 and unsure["number_missing"] == 1
    assert unsure["hits"] == 0.4 and unsure["distinct_sources"] == 0.5
    assert unsure["top_similarity"] == 1 and unsure["mean_similarity"] == 0.5
    grounded = gate_features(question, hits, "Caroline went on 14 July 2023.", vectors)
    assert grounded["draft_unsure"] == 0 and grounded["number_missing"] == 0
    assert grounded["draft_grounding"] > 0.5
    assert gate_features("def f(): pass", [], "", vectors[:2])["code_question"] == 1


def test_draft_naming_a_retrieved_file():
    hits = [memory("query building", "django/db/models/sql/query.py@8f4eaef6")]
    vectors = [np.array([1.0, 0.0])] * 3
    named = gate_features("Fix the bug", hits, "django/db/models/sql/query.py\ndjango/forms/a.py", vectors)
    assert named["draft_names_source"] == 0.5
    assert gate_features("Fix the bug", hits, "no paths here", vectors)["draft_names_source"] == 0


def test_logprobs_and_the_self_check_are_read_from_the_response():
    assert token_logprobs(response("a b", [{"token": "a", "logprob": -0.1},
                                           {"token": "b", "logprob": -2.0}])) == [-0.1, -2.0]
    assert token_logprobs(response("a")) == []
    check = response("Yes", [{"token": "Yes", "logprob": math.log(0.6), "top_logprobs": [
        {"token": "Yes", "logprob": math.log(0.6)}, {"token": " no", "logprob": math.log(0.2)},
        {"token": "Maybe", "logprob": math.log(0.1)}]}])
    assert yes_probability(check) == pytest.approx(0.75)
    assert yes_probability(response("ok")) == 0.5  # no logprobs: no opinion


async def test_draft_confidence_resamples_and_self_checks():
    calls = []

    async def chat(body):
        calls.append(body)
        if body.get("n"):  # three of four samples agree with the draft's first line
            return {"choices": [{"message": {"content": t}} for t in
                                ("14 July 2023", "14 July 2023.", "14 july 2023", "June")]}
        return response("yes", [{"token": "yes", "logprob": math.log(0.9), "top_logprobs": [
            {"token": "yes", "logprob": math.log(0.9)}, {"token": "no", "logprob": math.log(0.1)}]}])

    async def embed(texts):
        return [np.array([1.0, 0.0]) if "july" in t.lower() else np.array([0.0, 1.0]) for t in texts]

    body = {"messages": [{"role": "system", "content": "memory"}, {"role": "user", "content": "When?"}],
            "temperature": 0, "logprobs": True}
    draft = response("14 July 2023", [{"token": "14", "logprob": -0.1}, {"token": " July", "logprob": -1.0},
                                      {"token": " 2023", "logprob": -0.2}])
    f = await draft_confidence(chat, embed, body, draft)
    assert set(f) == set(CONFIDENCE_FEATURES)
    assert f["draft_logprob_mean"] == pytest.approx(-1.3 / 3) and f["draft_logprob_min"] == -1.0
    assert f["draft_unlikely_share"] == pytest.approx(1 / 3)
    assert f["sample_same_answer"] == 0.75 and f["sample_agreement"] == pytest.approx(0.75)
    assert f["self_check"] == pytest.approx(0.9)
    samples, check = sorted(calls, key=lambda b: "n" not in b)
    assert samples["n"] == SAMPLES and "logprobs" not in samples and samples["messages"] == body["messages"]
    assert check["max_tokens"] == 1 and check["messages"][:2] == body["messages"]
    assert check["messages"][2] == {"role": "assistant", "content": "14 July 2023"}


async def test_a_runaway_draft_keeps_the_probes_within_context():
    calls = []

    async def chat(body):
        calls.append(body)
        return response("x")

    async def embed(texts):
        return [np.array([1.0, 0.0]) for _ in texts]

    body = {"messages": [{"role": "user", "content": "Which file?"}]}
    await draft_confidence(chat, embed, body, response("sphinx/autodoc.py\n" * 4000))
    samples, check = sorted(calls, key=lambda b: "n" not in b)
    assert samples["max_tokens"] == SAMPLE_MAX_TOKENS
    assert len(check["messages"][1]["content"]) == CHECK_DRAFT_CHARS


def reasoned(text, reasoning, logprobs, completion_tokens=0):
    """A reasoning model's response: vLLM's logprobs cover the reasoning channel too."""
    r = response(text, logprobs)
    r["choices"][0]["message"]["reasoning"] = reasoning
    r["usage"] = {"completion_tokens": completion_tokens}
    return r


def test_after_reasoning_only_the_answer_is_read():
    tokens = [{"token": "The", "logprob": -3.0}, {"token": " file", "logprob": -2.0},
              {"token": "<|eom|>", "logprob": 0.0}, {"token": "a.py", "logprob": -0.1},
              {"token": "\nb.py", "logprob": -0.5}, {"token": "<|eot|>", "logprob": 0.0}]
    assert token_logprobs(reasoned("a.py\nb.py", "The file", tokens)) == [-0.1, -0.5]
    assert token_logprobs(reasoned("", "The file", tokens)) == []
    yes = {"token": "Yes", "logprob": math.log(0.8), "top_logprobs": [
        {"token": "Yes", "logprob": math.log(0.8)}, {"token": "No", "logprob": math.log(0.2)}]}
    check = reasoned("**Yes**", "It names the file",
                     [{"token": "It", "logprob": -1.0, "top_logprobs": []},
                      {"token": "**", "logprob": -0.1}, yes, {"token": "**", "logprob": -0.1}])
    assert yes_probability(check) == pytest.approx(0.8)


async def test_a_reasoning_draft_gives_its_probes_room_to_reason():
    calls = []

    async def chat(body):
        calls.append(body)
        return response("x")

    async def embed(texts):
        return [np.array([1.0, 0.0]) for _ in texts]

    body = {"messages": [{"role": "user", "content": "Which file?"}]}
    await draft_confidence(chat, embed, body, reasoned("a.py", "Let me think", [], 900))
    samples, check = sorted(calls, key=lambda b: "n" not in b)
    assert samples["max_tokens"] == 1800 and check["max_tokens"] == REASONING_MAX_TOKENS


def test_prompt_estimates_per_route():
    hits, pages = [memory("s", tokens=10)] * 2, [memory("p", tokens=500)] * 2
    tokens = estimated_prompt_tokens("x" * 40, hits, pages)
    assert tokens[Route.LOCAL_MEMORY] == 0
    assert tokens[Route.ESCALATE_COMPACT] == 10 + 40 + 20
    assert tokens[Route.ESCALATE_RAW] == 10 + 40 + 1000


def test_gate_trades_accuracy_for_cost(tmp_path):
    gate = constant_gate(0.3, 0.8)
    p = gate.probabilities({f: 0.0 for f in FEATURES})
    assert p == pytest.approx({"local_memory": 0.3, "escalate_compact": 0.8})
    costs = gate.costs({"local_memory": 0, "escalate_compact": 500}, Ledger(None, 0, 1.0, 4.0).cost)
    assert costs == pytest.approx({"local_memory": 0.0, "escalate_compact": (500 + 4000) / 1e6})
    assert gate.decide(p, costs) == Route.ESCALATE_COMPACT        # lam 0: accuracy only
    assert gate.decide(p, costs, lam=1e3) == Route.LOCAL_MEMORY   # 0.5 better isn't worth $0.0045
    gate.save(tmp_path / "gate.json")
    assert LearnedGate.load(tmp_path / "gate.json") == gate
    assert not gate.needs_confidence
    assert constant_gate(0.3, 0.8, features=FEATURES + CONFIDENCE_FEATURES).needs_confidence


def test_fit_logistic_finds_the_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    w = fit_logistic(x, (x[:, 0] > 0).astype(float))
    assert w[0] > 1 and abs(w[1]) < 0.5
    assert np.isfinite(fit_logistic(x, np.ones(400))).all()


def test_folds_keep_each_team_together():
    groups = ["a"] * 5 + ["b"] * 3 + ["c"] * 3 + ["d"]
    fold = folds(groups, k=3)
    assert all(len({f for g2, f in zip(groups, fold) if g2 == g}) == 1 for g in set(groups))
    assert set(fold) == {0, 1, 2}


def test_auc_and_operating_point():
    assert auc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert auc([0.5, 0.5], [True, False]) == 0.5
    assert auc([0.5], [True]) is None
    sweep = [{"lam": lam, "a": {"accuracy": acc, "muse_token_share": share}}
             for lam, acc, share in ((1, 0.9, 0.3), (10, 0.8, 0.09), (20, 0.8, 0.05), (99, 0.5, 0.0))]
    assert operating_point(sweep, "a", 0.10)["lam"] == 20  # the most accurate within budget; tie: cheaper


def test_code_memory_escalates_compact_and_other_memory_raw():
    code = [memory("x", "django/db/models/sql/query.py@8f4eaef6")] * 3
    talk = [memory("x", "locomo:conv-26:session_2")] * 2
    assert escalation_route(code) == Route.ESCALATE_COMPACT
    assert escalation_route(code[:2] + talk) == Route.ESCALATE_COMPACT  # mostly code
    assert escalation_route(code[:1] + talk) == Route.ESCALATE_RAW
    assert escalation_route([]) == Route.ESCALATE_RAW


def test_cascade_keeps_confident_drafts(tmp_path):
    rows = [{"features": {f: v for f in CONFIDENCE_FEATURES}, "correct": {"local_memory": v > 0.5}}
            for v in (0.1, 0.2, 0.3, 0.7, 0.8, 0.9) * 10]
    gate = train_cascade(rows, list(CONFIDENCE_FEATURES), threshold=0.5)
    sure, unsure = ({f: v for f in CONFIDENCE_FEATURES} for v in (0.9, 0.1))
    assert gate.p_right(sure) > 0.5 > gate.p_right(unsure)
    talk = [memory("x", "notes.md")]
    assert gate.decide(gate.p_right(sure), talk) == Route.LOCAL_MEMORY
    assert gate.decide(gate.p_right(unsure), talk) == Route.ESCALATE_RAW
    assert gate.needs_confidence
    gate.save(tmp_path / "cascade.json")
    assert load_gate(tmp_path / "cascade.json") == gate
    constant_gate(0.3, 0.8).save(tmp_path / "routes.json")
    assert isinstance(load_gate(tmp_path / "routes.json"), LearnedGate)


def test_cascade_sweep_picks_the_most_accurate_threshold_within_budget():
    rows = [{"p_draft": p, "escalate_to": "escalate_raw"} for p in (0.2, 0.6)]
    assert cascade_choose(rows, 0.5) == ["escalate_raw", "local_memory"]
    sweep = [{"threshold": t, "a": {"accuracy": acc, "muse_token_share": share}}
             for t, acc, share in ((0.0, 0.5, 0.0), (0.3, 0.8, 0.05), (0.4, 0.8, 0.08), (0.9, 0.9, 0.3))]
    assert cascade_point(sweep, "a", 0.10)["threshold"] == 0.3  # tie: the cheaper, lower threshold

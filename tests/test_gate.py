import math
from uuid import uuid4

import numpy as np
import pytest

from eval.learn_gate import fit_logistic, folds
from gateway.accounting import Ledger
from gateway.gate import FEATURES, LearnedGate, Route, estimated_prompt_tokens, gate_features
from memory.store import Memory


def memory(content, source="notes", tokens=10):
    return Memory(id=uuid4(), owner_id=None, kind="summary", parent_id=None, source=source,
                  content=content, keywords=[], token_count=tokens)


def constant_gate(p_local, p_escalate, lam=0.0):
    """A gate whose P(right) per route ignores the features."""
    logit = lambda p: math.log(p / (1 - p))  # noqa: E731
    zeros = [0.0] * len(FEATURES)
    return LearnedGate(routes=["local_memory", "escalate_compact"], mean=zeros, scale=[1.0] * len(FEATURES),
                       weights={"local_memory": [*zeros, logit(p_local)],
                                "escalate_compact": [*zeros, logit(p_escalate)]},
                       output_tokens=1000, lam=lam)


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

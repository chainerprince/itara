"""Decide, per turn, whether to answer locally or escalate to the closed model.

This is the rule-based baseline. The learned gate (trained on escalation regret)
replaces `rule_gate` later and keeps the same inputs and outputs.
"""

from __future__ import annotations

from enum import StrEnum

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

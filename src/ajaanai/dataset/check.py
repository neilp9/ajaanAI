"""Responsiveness check: does each teacher reply answer the caller's line just before it?

Synthesized replies are his sentences from one passage, which makes it easy to build a conversation
that merely walks through the passage while the caller talks past it. Trained on that, the model
learned to continue its talk instead of answering (eval interactivity 2.08 vs 3.74 for the base).
A cheap model (CHECK_MODEL) labels every reply; the build cuts each conversation at its first
unresponsive reply.

Only the caller / teacher lines are sent (no system prompt or passage), keeping the check cheap.
Runs through the same batch/sync machinery as synthesis, with its own resumable store.
"""

from __future__ import annotations

from ..llm import Request

CHECK_SYSTEM = """You check training conversations for an AI meant to talk with people the way a \
good teacher does: each reply responds to what the person just said.

For each AI reply, in order, answer true if it responds to the person's line immediately before it \
— it answers their question, addresses what they said, or follows up on their answer to its \
question. Answer false if it ignores that line, carries on an earlier thread the person has moved \
away from, or only makes sense as a continuation of the AI's own previous reply.
A reply can be brief or indirect and still be responsive; judge relevance, not style."""

CHECK_SCHEMA = {
    "type": "object",
    "properties": {"responsive": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["responsive"],
    "additionalProperties": False,
}


def render_turns(turns: list[dict]) -> str:
    lines = [f"{'Person' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in turns]
    n = sum(m["role"] == "assistant" for m in turns)
    return "\n\n".join(lines) + f"\n\nLabel each of the {n} AI replies."


def check_params(model: str, turns: list[dict]) -> Request:
    return Request(model=model, system=CHECK_SYSTEM, user=render_turns(turns), schema=CHECK_SCHEMA, max_tokens=512)


def responsive_prefix(turns: list[dict], labels: list[bool] | None) -> tuple[list[dict], int]:
    """Turns up to (not including) the first exchange whose reply was labelled unresponsive, and
    how many exchanges were cut. Missing or malformed labels keep the conversation as it is."""
    n_replies = sum(m["role"] == "assistant" for m in turns)
    if labels is None or len(labels) != n_replies or all(labels):
        return turns, 0
    first_bad = labels.index(False)
    return turns[: 2 * first_bad], n_replies - first_bad

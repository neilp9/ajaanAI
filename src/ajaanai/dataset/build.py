"""Assemble provider-neutral multi-turn chat JSONL ({"messages": [...]}) from synthesized conversations.

Each example is one conversation: the system prompt for its channel (voice call or text chat,
see chunk.passage_channel), then alternating caller / teacher turns.
Every teacher turn is his own sentences, kept short (1-3 sentences) unless the caller asked for
more; the one exception is an optional short question back to the caller (see synth.py), which
DATASET_ASK_BACK can leave out.

Example types:
  conversation           scripted call built on a passage of his talks or writings
  qa                     a real retreat Q&A exchange, his answer spread over a few turns
  *_grounded             same, with the source passage given as a reference (teaches using RAG context)

A turn that breaks the rules (too long, reused sentences, a first reply that leans on missing
context) ends the conversation just before it, so every example stays coherent.
Split is by document, so no eval passage's text leaks into training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .chunk import Passage, passage_channel, split_sentences, stable_bucket
from .persona import RAG_PREAMBLE, system_prompt

MAX_TURN_SENTENCES, MAX_TURN_WORDS = 4, 70  # an ordinary reply
MAX_LONG_SENTENCES, MAX_LONG_WORDS = 7, 140  # when the caller asked him to say more
MAX_ASK_BACK_WORDS = 14

# The first reply of a call is heard with nothing before it: reject openings that lean on a dropped
# sentence, and openings that quote someone else (the suttas, other teachers) rather than his words.
_DANGLING_START = re.compile(
    r"^(?:[\"“‘'(]|(?:So|And|But|Or|Because|Which|Then|Also|Thus|Hence|Otherwise|Instead|Yet|Still|"
    r"In this way|In that way|That way|This way|In other words|It[’']s in this way|"
    r"This is why|That[’']s why|This is how|That[’']s how|At the same time|In particular|"
    r"Other times|All these|All of these|All this|All of this|Of course|Again|As a result)\b)")

@dataclass
class BuildStats:
    train: int = 0
    eval: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_channel: dict[str, int] = field(default_factory=dict)
    skipped_unusable: int = 0
    assistant_turns: int = 0
    ask_backs: int = 0
    turns_dropped: int = 0


def _reply(sentences: list[str], ids: list[int], *, longer: bool, first: bool, used: set[int]) -> str | None:
    """His sentences for one turn, or None if the turn breaks the rules."""
    picked = sorted({i for i in ids if 0 <= i < len(sentences)})
    max_s, max_w = (MAX_LONG_SENTENCES, MAX_LONG_WORDS) if longer else (MAX_TURN_SENTENCES, MAX_TURN_WORDS)
    if not 1 <= len(picked) <= max_s or used.intersection(picked):
        return None
    # Dropped, not patched: borrowing the sentence before a leaning opener tends to bring in a
    # dangling pronoun or the tail of a quotation instead.
    if first and _DANGLING_START.match(sentences[picked[0]]):
        return None
    text = " ".join(sentences[i] for i in picked)
    if len(text.split()) > max_w:
        return None
    used.update(picked)
    return text


def _ask_back(text: str) -> str | None:
    text = " ".join(text.split())
    return text if text.endswith("?") and len(text.split()) <= MAX_ASK_BACK_WORDS else None


def conversation_turns(p: Passage, convo: dict, *, ask_back: bool = True) -> tuple[list[dict], int, int]:
    """(user/assistant messages, ask-backs kept, exchanges dropped) for one synthesized conversation."""
    sentences = split_sentences(p.text)
    exchanges = convo.get("exchanges", [])
    msgs: list[dict] = []
    used: set[int] = set()
    n_ask = 0
    for n, ex in enumerate(exchanges):
        caller = p.question if (n == 0 and p.question) else " ".join(ex.get("caller", "").split())
        reply = _reply(sentences, ex.get("sentences", []), longer=bool(ex.get("longer")), first=n == 0, used=used)
        if not caller or reply is None:
            return msgs, n_ask, len(exchanges) - n
        msgs += [{"role": "user", "content": caller}, {"role": "assistant", "content": reply}]
        if raw := ex.get("ask_back", "").strip():
            question = _ask_back(raw) if ask_back else None
            if question is None:
                # the caller's next line answers this question, so the call can't go on without it
                return msgs, n_ask, len(exchanges) - n - 1
            msgs[-1]["content"] += " " + question
            n_ask += 1
    return msgs, n_ask, 0


def build_examples(passages: list[Passage], synth: dict[str, dict], eval_fraction: float, *,
                   ask_back: bool = True) -> tuple[list[dict], list[dict], list[dict], BuildStats]:
    """Returns (train, eval, eval_questions, stats)."""
    stats = BuildStats()
    train, evals, eval_qs = [], [], []
    seen_q: set[str] = set()

    for p in passages:
        out = synth.get(p.id)
        if out is None:
            continue
        if not out.get("usable", False):
            stats.skipped_unusable += 1
            continue
        is_eval = stable_bucket(p.doc_id) < eval_fraction * 10_000
        channel = passage_channel(p)
        for convo in out.get("conversations", []):
            turns, n_ask, dropped = conversation_turns(p, convo, ask_back=ask_back)
            stats.turns_dropped += dropped
            if not turns or turns[0]["content"].lower() in seen_q:
                continue
            q = turns[0]["content"]
            seen_q.add(q.lower())
            if is_eval:
                eval_qs.append({"question": q, "reference": p.text, "doc_id": p.doc_id,
                                "passage_id": p.id, "title": p.title, "channel": channel})
            grounded = stable_bucket(p.id + q) % 10 < 3
            system = system_prompt(channel) + ("\n\n" + RAG_PREAMBLE + p.text if grounded else "")
            kind = ("qa" if p.kind == "qa" else "conversation") + ("_grounded" if grounded else "")
            (evals if is_eval else train).append({"messages": [{"role": "system", "content": system}, *turns]})
            stats.by_type[kind] = stats.by_type.get(kind, 0) + 1
            stats.by_channel[channel] = stats.by_channel.get(channel, 0) + 1
            stats.assistant_turns += len(turns) // 2
            stats.ask_backs += n_ask

    stats.train, stats.eval = len(train), len(evals)
    return train, evals, eval_qs, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

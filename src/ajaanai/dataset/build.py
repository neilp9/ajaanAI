"""Assemble provider-neutral multi-turn chat JSONL ({"messages": [...]}) from synthesized conversations.

Each example is one conversation: the system prompt for its channel (voice call or text chat,
see chunk.passage_channel), then alternating caller / teacher turns.
Every teacher turn is his own sentences, kept short (1-3 sentences) unless the caller asked for
more. The only other words on his side are optional model-written extras (see synth.py): a brief
acknowledgement before his sentences (DATASET_ACKNOWLEDGE) and a short question back after them
(DATASET_ASK_BACK). With DATASET_REPLY_MODE=shaped, teacher turns are instead written by the synth model in his
voice from the passage (`shaped=True` here); the same length and opening rules apply.

Example types:
  conversation           scripted call built on a passage of his talks or writings
  qa                     a real retreat Q&A exchange, his answer spread over a few turns
  *_grounded             same, with the source passage given as a reference (teaches using RAG context)

A turn that breaks the rules (too long, reused sentences, a reply opening that leans on missing
context) ends the conversation just before it, so every example stays coherent. So does a reply
the responsiveness check (check.py) labels as not answering the caller's line.
Split is by document, so no eval passage's text leaks into training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .check import responsive_prefix
from .chunk import Passage, passage_channel, split_sentences, stable_bucket
from .persona import RAG_PREAMBLE, system_prompt

MAX_TURN_SENTENCES, MAX_TURN_WORDS = 4, 70  # an ordinary reply
MAX_LONG_SENTENCES, MAX_LONG_WORDS = 7, 140  # when the caller asked him to say more
MAX_ASK_BACK_WORDS = 14
MAX_ACK_WORDS = 8

# Every reply is heard as a direct answer to the caller: reject openings that lean on a sentence
# that isn't there, and openings that quote someone else (the suttas, other teachers). Allowing
# them after the first turn taught the model to continue its talk instead of answering (37% of its
# later eval replies opened with "So"/"And"/"But"/"Or").
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
    acknowledgements: int = 0
    turns_dropped: int = 0  # cut for breaking the reply rules
    turns_not_responsive: int = 0  # cut by the responsiveness check


@dataclass
class Candidate:
    """One synthesized conversation that passed the reply rules, before the responsiveness check."""
    conv_id: str  # "<passage id>#c<n>"
    passage: Passage
    turns: list[dict]
    ask_backs: list[bool]  # per teacher reply: carries a model-written question back
    acknowledgements: list[bool]  # per teacher reply: opens with a model-written acknowledgement


def _reply(sentences: list[str], ids: list[int], *, longer: bool, used: set[int]) -> str | None:
    """His sentences for one turn, or None if the turn breaks the rules."""
    picked = sorted({i for i in ids if 0 <= i < len(sentences)})
    max_s, max_w = (MAX_LONG_SENTENCES, MAX_LONG_WORDS) if longer else (MAX_TURN_SENTENCES, MAX_TURN_WORDS)
    if not 1 <= len(picked) <= max_s or used.intersection(picked):
        return None
    # Dropped, not patched: borrowing the sentence before a leaning opener tends to bring in a
    # dangling pronoun or the tail of a quotation instead.
    if _DANGLING_START.match(sentences[picked[0]]):
        return None
    text = " ".join(sentences[i] for i in picked)
    if len(text.split()) > max_w:
        return None
    used.update(picked)
    return text


_LIST = re.compile(r"^(?:[-*]\s|\d+[.)]\s)|\*\*|•")  # list or markdown formatting


def _shaped_reply(text: str, *, longer: bool) -> str | None:
    """A written reply (shaped mode), or None if it breaks the rules. Same limits as a verbatim reply,
    plus room for the short question back that verbatim mode adds separately."""
    text = " ".join(text.split())
    max_s, max_w = (MAX_LONG_SENTENCES, MAX_LONG_WORDS) if longer else (MAX_TURN_SENTENCES, MAX_TURN_WORDS)
    if not text or len(split_sentences(text)) > max_s + 1 or len(text.split()) > max_w + MAX_ASK_BACK_WORDS:
        return None
    if _DANGLING_START.match(text) or _LIST.search(text):
        return None
    return text


def _ask_back(text: str) -> str | None:
    text = " ".join(text.split())
    return text if text.endswith("?") and len(text.split()) <= MAX_ASK_BACK_WORDS else None


def _acknowledgement(text: str) -> str | None:
    text = " ".join(text.split())
    if not text or text.endswith("?") or len(text.split()) > MAX_ACK_WORDS:
        return None
    return text if text[-1] in ".!" else text + "."


def conversation_turns(p: Passage, convo: dict, *, ask_back: bool = True, acknowledge: bool = True,
                       shaped: bool = False) -> tuple[list[dict], list[bool], list[bool], int]:
    """(user/assistant messages, per-reply ask-back flags, per-reply acknowledgement flags,
    exchanges dropped) for one synthesized conversation. In shaped mode a reply's ask-back flag
    means it ends on a question; `ask_back` / `acknowledge` don't apply (they're part of the text)."""
    sentences = split_sentences(p.text)
    exchanges = convo.get("exchanges", [])
    msgs: list[dict] = []
    used: set[int] = set()
    asks: list[bool] = []
    acks: list[bool] = []
    for n, ex in enumerate(exchanges):
        caller = p.question if (n == 0 and p.question) else " ".join(ex.get("caller", "").split())
        if shaped:
            reply = _shaped_reply(ex.get("reply", ""), longer=bool(ex.get("longer")))
            if not caller or reply is None:
                return msgs, asks, acks, len(exchanges) - n
            msgs += [{"role": "user", "content": caller}, {"role": "assistant", "content": reply}]
            asks.append(reply.endswith("?"))
            acks.append(False)
            continue
        reply = _reply(sentences, ex.get("sentences", []), longer=bool(ex.get("longer")), used=used)
        if not caller or reply is None:
            return msgs, asks, acks, len(exchanges) - n
        # An acknowledgement only reflects back what the caller said, so dropping it (when disabled
        # or malformed) never breaks the conversation.
        ack = _acknowledgement(ex.get("acknowledge", "")) if acknowledge else None
        if ack:
            reply = f"{ack} {reply}"
        acks.append(bool(ack))
        asks.append(False)
        msgs += [{"role": "user", "content": caller}, {"role": "assistant", "content": reply}]
        if raw := ex.get("ask_back", "").strip():
            question = _ask_back(raw) if ask_back else None
            if question is None:
                # the caller's next line answers this question, so the call can't go on without it
                return msgs, asks, acks, len(exchanges) - n - 1
            msgs[-1]["content"] += " " + question
            asks[-1] = True
    return msgs, asks, acks, 0


def conversation_candidates(passages: list[Passage], synth: dict[str, dict], *, ask_back: bool = True,
                            acknowledge: bool = True, shaped: bool = False,
                            stats: BuildStats | None = None) -> list[Candidate]:
    """Every synthesized conversation that survives the reply rules, deduplicated by opening line."""
    stats = stats if stats is not None else BuildStats()
    out: list[Candidate] = []
    seen_q: set[str] = set()
    for p in passages:
        result = synth.get(p.id)
        if result is None:
            continue
        if not result.get("usable", False):
            stats.skipped_unusable += 1
            continue
        for k, convo in enumerate(result.get("conversations", [])):
            turns, asks, acks, dropped = conversation_turns(p, convo, ask_back=ask_back, acknowledge=acknowledge,
                                                            shaped=shaped)
            stats.turns_dropped += dropped
            if not turns or turns[0]["content"].lower() in seen_q:
                continue
            seen_q.add(turns[0]["content"].lower())
            out.append(Candidate(f"{p.id}#c{k}", p, turns, asks, acks))
    return out


def build_examples(passages: list[Passage], synth: dict[str, dict], eval_fraction: float, *,
                   ask_back: bool = True, acknowledge: bool = True, shaped: bool = False,
                   responsive: dict[str, list[bool]] | None = None,
                   ) -> tuple[list[dict], list[dict], list[dict], BuildStats]:
    """Returns (train, eval, eval_questions, stats). `responsive` maps conversation id -> one label
    per teacher reply from the responsiveness check; each conversation is cut at its first False."""
    stats = BuildStats()
    train, evals, eval_qs = [], [], []
    for c in conversation_candidates(passages, synth, ask_back=ask_back, acknowledge=acknowledge, shaped=shaped,
                                     stats=stats):
        p = c.passage
        turns, cut = responsive_prefix(c.turns, (responsive or {}).get(c.conv_id))
        stats.turns_not_responsive += cut
        if not turns:
            continue
        is_eval = stable_bucket(p.doc_id) < eval_fraction * 10_000
        channel = passage_channel(p)
        q = turns[0]["content"]
        if is_eval:
            eval_qs.append({"question": q, "reference": p.text, "doc_id": p.doc_id,
                            "passage_id": p.id, "title": p.title, "channel": channel})
        grounded = stable_bucket(p.id + q) % 10 < 3
        system = system_prompt(channel) + ("\n\n" + RAG_PREAMBLE + p.text if grounded else "")
        kind = ("qa" if p.kind == "qa" else "conversation") + ("_grounded" if grounded else "")
        (evals if is_eval else train).append({"messages": [{"role": "system", "content": system}, *turns]})
        stats.by_type[kind] = stats.by_type.get(kind, 0) + 1
        stats.by_channel[channel] = stats.by_channel.get(channel, 0) + 1
        n_kept = len(turns) // 2
        stats.assistant_turns += n_kept
        stats.ask_backs += sum(c.ask_backs[:n_kept])
        stats.acknowledgements += sum(c.acknowledgements[:n_kept])

    stats.train, stats.eval = len(train), len(evals)
    return train, evals, eval_qs, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

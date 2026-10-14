"""Conversation synthesis with Claude.

For each passage, Claude scripts a short conversation — a voice call or a text chat, per
`passage_channel` — whose caller lines are Claude's (styled for that channel), and
each of the teacher's turns is a few of *his own sentences*, picked by number to answer the caller's
line (about a third of caller lines go off-script, so replies learn to respond rather than just
continue the passage). The only teacher-side words Claude may write are an optional brief
acknowledgement before his sentences (`acknowledge`) and an optional short question back after them
(`ask_back`); both are kept apart in the results so the build can include or drop them
(DATASET_ACKNOWLEDGE, DATASET_ASK_BACK).

Real retreat Q&A exchanges (passage.kind == "qa") are handled the same way: the real question opens
the call and his real answer is spread over a few turns with brief caller follow-ups in between.

Bulk runs go through the Message Batches API (50% cheaper, resumable); `--sample` runs
synchronously for a quick look. `run_batch` / `run_sync` are shared with the responsiveness check
(check.py).

Stored results are matched by item id *and* a hash of the request's messages (`input_sha`), so an
item whose input changed — e.g. a talk whose ASR text was replaced by the site transcript, which
keeps its passage ids but renumbers the sentences — is redone instead of reusing sentence numbers
that now point elsewhere. The model and system prompt are not hashed: changing them doesn't redo
old results. Rows saved before hashing (no `input_sha`) count as current.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .chunk import Passage, passage_channel, split_sentences

log = logging.getLogger(__name__)

_INTRO = """You help build a training set for an assistant that talks with people — over voice calls \
and text chat — in the words of Thanissaro Bhikkhu, a Theravada monk. The conversations are real \
back-and-forth, not lectures: he makes one point at a time, briefly, and the caller responds."""

_RULES = """Each exchange is the caller's line followed by his reply.

Caller lines (you write these): usually under 25 words. Vary them: a follow-up, pushback or doubt, \
the caller's own situation, "what do you mean by…?", a request for an example, or simply taking in \
what he said. Not every line needs to be a question. About one line in three should go off-script — \
a personal detail, a tangent, a misunderstanding, or a question the next part of the text doesn't \
answer — the way real callers do. Don't open them all the same way, don't lean on the word \
"actually", and never mention "the passage", "the talk" or "the book".
Write them for the channel you're given, with a light touch (most lines should still read cleanly):
- voice call: transcribed speech — spoken phrasing, sometimes run-on or with an "um"; now and then \
a Pali term spelled the way speech recognition mishears it (e.g. "meta" for mettā, "jana" for jhāna).
- text chat: typed messages — often shorter and more casual, sometimes lowercase or with a small \
typo; Pali terms spelled however a layperson might type them.
His replies are the same on both channels.

His replies: `sentences` is the numbers of 1-3 of his sentences (about 60 words at most), read in \
order, that answer the caller's line — what they just said, not merely the next part of the text. \
Pick them from anywhere in the passage; consecutive sentences read best. Use each sentence at most \
once in a conversation. Every reply is heard as a direct answer, so its first sentence must stand on \
its own: never start on a connective or a back-reference ("So", "And", "But", "Or", "Because", \
"In this way", "This is why") or on an incomplete sentence. Don't use sentences that quote the \
suttas or other teachers. If no sentences answer a caller line well, write a different caller line.
Before finalizing each reply, read the first word of its first sentence: if it is "So", "And", \
"But", "Or", "Then", "Because", "This", "That", "Of course" or similar, choose different sentences \
(often the sentence just before is the real start of the thought) or change the caller line.
Only when the caller asks him to say more or go deeper, set `longer` and use up to 6 sentences.

`acknowledge`: when the caller has said something specific — a detail, a feeling, an answer to his \
question — you may open his reply with a brief acknowledgement in his plain manner (at most 8 \
words) showing he heard that particular line. Vary the form; don't fall into a template like \
"X, yes." Possibilities include a plain "Right." or "Okay.", taking up their words ("Two in the \
morning is a hard hour."), or a short observation ("That's the mind looking for work."). Never brush \
off what they shared, and don't lean on calling things "common"; no praise, no therapy-speak. Use it \
on about one reply in four. Otherwise "".

`ask_back`: on about one reply in four, add one short question back to the caller (at most 12 \
words) in his plain, direct manner — usually turning them toward their own experience, e.g. "What \
do you notice when that happens?". The caller's next line must respond to it. Otherwise leave it "".
Never write anything else on his side."""

SYSTEM = f"""{_INTRO}

You will see one passage from his talks or writings, with each sentence numbered. Write 1-2 \
conversations (2 if the passage covers two distinct points), each 3-6 exchanges, that this passage \
can carry. Callers are practitioners, beginners, or people going through something difficult.

{_RULES}

Mark the passage unusable if it is chanting, logistics, a fragment, mostly a quotation of someone \
else, or otherwise not a teaching that could answer a caller."""

SYSTEM_QA = f"""{_INTRO}

You will see a real question someone asked him at a retreat, and his real answer with each \
sentence numbered. Turn it into one conversation of 2-5 exchanges: the first caller line is \
the real question (copy it), and his answer is spread over the replies, with caller lines in between \
that the next part genuinely answers. Keep his answer roughly in order; you may leave out sentences \
that don't fit a one-to-one conversation (references to earlier sessions, retreat logistics).

{_RULES}

Mark it unusable if the question is about logistics or the answer can't stand without the retreat \
context."""

SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exchanges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "caller": {"type": "string"},
                                "sentences": {"type": "array", "items": {"type": "integer"}},
                                "longer": {"type": "boolean"},
                                "acknowledge": {"type": "string"},
                                "ask_back": {"type": "string"},
                            },
                            "required": ["caller", "sentences", "longer", "acknowledge", "ask_back"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["exchanges"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["usable", "topics", "conversations"],
    "additionalProperties": False,
}


def render(p: Passage) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(split_sentences(p.text)))
    channel = "Channel: " + ("text chat" if passage_channel(p) == "text" else "voice call")
    if p.question is not None:
        return f"{channel}\n\nQuestion: {p.question}\n\nHis answer:\n{numbered}\n\nWrite the conversation."
    return f"{channel}\n\nTitle: {p.title}\n\n{numbered}\n\nWrite 1-2 conversations."


def _params(model: str, p: Passage) -> dict:
    return dict(
        model=model,
        max_tokens=4096,
        system=SYSTEM_QA if p.question is not None else SYSTEM,
        messages=[{"role": "user", "content": render(p)}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )


def _parse(message) -> dict | None:
    if message.stop_reason != "end_turn":
        return None
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def input_sha(params: dict) -> str:
    """Hash of what the model sees about the item (its messages, not the model or system prompt)."""
    return hashlib.sha256(json.dumps(params["messages"], sort_keys=True).encode()).hexdigest()[:16]


def _current(row: dict, sha: str | None) -> bool:
    return row.get("input_sha") in (None, sha)


class SynthStore:
    """Append-only results file + batch bookkeeping so runs can resume after interruption.
    `prefix` keeps separate stages (synthesis, the responsiveness check) apart in one directory.
    The latest row for an id wins."""

    def __init__(self, root: Path, prefix: str = "synth"):
        root.mkdir(parents=True, exist_ok=True)
        self.results = root / f"{prefix}_results.jsonl"
        self.batches = root / f"{prefix}_batches.json"
        self.idmap = root / f"{prefix}_idmap.json"

    def todo(self, items: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """Items with no stored result, or whose stored result was made from a different input."""
        rows = self.load()
        return [(i, params) for i, params in items if i not in rows or not _current(rows[i], input_sha(params))]

    def append(self, item_id: str, out: dict, sha: str | None = None) -> None:
        with self.results.open("a") as f:
            f.write(json.dumps({"passage_id": item_id, "input_sha": sha, **out}) + "\n")

    def load(self, expected: dict[str, str] | None = None) -> dict[str, dict]:
        """Latest result per id. With `expected` (id -> input_sha), results made from a different
        input are left out, so a failed redo never falls back to a stale result."""
        if not self.results.exists():
            return {}
        rows = {(r := json.loads(line))["passage_id"]: r for line in self.results.open()}
        if expected is not None:
            rows = {i: r for i, r in rows.items() if _current(r, expected.get(i))}
        return rows

    def pending_batches(self) -> list[str]:
        return json.loads(self.batches.read_text()) if self.batches.exists() else []

    def set_pending_batches(self, ids: list[str]) -> None:
        self.batches.write_text(json.dumps(ids))


def run_sync(client: anthropic.Anthropic, items: list[tuple[str, dict]], store: SynthStore) -> int:
    """One request at a time for (id, params) items not yet in the store. Returns results stored."""
    n = 0
    for item_id, params in store.todo(items):
        out = _parse(client.messages.create(**params))
        if out is not None:
            store.append(item_id, out, input_sha(params))
            n += 1
    return n


def run_batch(client: anthropic.Anthropic, items: list[tuple[str, dict]], store: SynthStore,
              poll_s: int = 60, max_per_batch: int = 50_000) -> int:
    """Submit (or resume) batches for (id, params) items not yet stored, wait, store results."""
    pending = store.pending_batches()
    if not pending:
        todo = store.todo(items)
        for k in range(0, len(todo), max_per_batch):
            part = todo[k : k + max_per_batch]
            batch = client.messages.batches.create(requests=[
                Request(custom_id=f"p{j}", params=MessageCreateParamsNonStreaming(**params))
                for j, (_, params) in enumerate(part, start=k)
            ])
            pending.append(batch.id)
            log.info("submitted batch %s (%d requests)", batch.id, len(part))
        store.set_pending_batches(pending)
        # custom_id -> [item id, input hash] (custom ids must be short)
        store.idmap.write_text(json.dumps({f"p{j}": [i, input_sha(params)] for j, (i, params) in enumerate(todo)}))

    idmap = json.loads(store.idmap.read_text()) if store.idmap.exists() else {}
    stored = 0
    for batch_id in list(pending):
        while (b := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
            log.info("batch %s: %s (processing=%d)", batch_id, b.processing_status, b.request_counts.processing)
            time.sleep(poll_s)
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                continue
            out = _parse(result.result.message)
            if out is not None and result.custom_id in idmap:
                item_id, sha = idmap[result.custom_id]
                store.append(item_id, out, sha)
                stored += 1
        pending.remove(batch_id)
        store.set_pending_batches(pending)
    return stored


def synth_sync(client: anthropic.Anthropic, model: str, passages: list[Passage], store: SynthStore) -> int:
    return run_sync(client, [(p.id, _params(model, p)) for p in passages], store)


def synth_batch(client: anthropic.Anthropic, model: str, passages: list[Passage], store: SynthStore,
                poll_s: int = 60, max_per_batch: int = 50_000) -> int:
    return run_batch(client, [(p.id, _params(model, p)) for p in passages], store, poll_s, max_per_batch)

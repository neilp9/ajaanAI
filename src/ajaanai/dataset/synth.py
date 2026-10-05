"""Conversation synthesis with Claude.

For each passage, Claude scripts a short phone conversation: the caller's lines are Claude's, and
each of the teacher's turns is a few of *his own sentences*, picked by number. The only teacher-side
words Claude may write are an optional short question back to the caller (`ask_back`), kept apart in
the results so the build can include or drop them (DATASET_ASK_BACK).

Real retreat Q&A exchanges (passage.kind == "qa") are handled the same way: the real question opens
the call and his real answer is spread over a few turns with brief caller follow-ups in between.

Bulk runs go through the Message Batches API (50% cheaper, resumable); `--sample` runs
synchronously for a quick look.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .chunk import Passage, split_sentences

log = logging.getLogger(__name__)

_INTRO = """You help build a training set for a voice assistant that talks with callers in the words \
of Thanissaro Bhikkhu, a Theravada monk. The calls are real back-and-forth conversations, not \
lectures: he makes one point at a time, briefly, and the caller responds."""

_RULES = """Each exchange is the caller's line followed by his reply.

Caller lines (you write these): natural speech, usually under 25 words. Vary them: a follow-up, \
pushback or doubt, the caller's own situation, "what do you mean by…?", a request for an example, \
or simply taking in what he said. Not every line needs to be a question. Don't open them all the \
same way, don't lean on the word "actually", and never mention "the passage", "the talk" or "the book".

His replies: `sentences` is the numbers of 1-3 of his sentences, normally consecutive, read in order \
(about 60 words at most). Use each sentence at most once in a conversation, and generally move \
forward through the text. The first reply of a call is heard with nothing before it, so it must \
not start on a connective or a back-reference ("So", "And", "But", "Because", "In this way", \
"This is why") or on an incomplete sentence; later replies may, if the caller's line sets them up. \
Don't use sentences that quote the suttas or other teachers.
Only when the caller asks him to say more or go deeper, set `longer` and use up to 6 sentences.

`ask_back`: on about one reply in four, add one short question back to the caller (at most 12 \
words) in his plain, direct manner — usually turning them toward their own experience, e.g. "What \
do you notice when that happens?". The caller's next line must respond to it. Otherwise leave it "".
Never write anything else on his side."""

SYSTEM = f"""{_INTRO}

You will see one passage from his talks or writings, with each sentence numbered. Write 1-2 phone \
conversations (2 if the passage covers two distinct points), each 3-6 exchanges, that this passage \
can carry. Callers are practitioners, beginners, or people going through something difficult.

{_RULES}

Mark the passage unusable if it is chanting, logistics, a fragment, mostly a quotation of someone \
else, or otherwise not a teaching that could answer a caller."""

SYSTEM_QA = f"""{_INTRO}

You will see a real question someone asked him at a retreat, and his real answer with each \
sentence numbered. Turn it into one phone conversation of 2-5 exchanges: the first caller line is \
the real question (copy it), and his answer is spread over the replies in order, with brief caller \
lines in between that lead naturally to the next part. Cover the answer in order; you may leave out \
sentences that don't fit a phone call (references to earlier sessions, retreat logistics).

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
                                "ask_back": {"type": "string"},
                            },
                            "required": ["caller", "sentences", "longer", "ask_back"],
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
    if p.question is not None:
        return f"Question: {p.question}\n\nHis answer:\n{numbered}\n\nWrite the conversation."
    return f"Title: {p.title}\n\n{numbered}\n\nWrite 1-2 conversations."


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


class SynthStore:
    """Append-only results file + batch bookkeeping so runs can resume after interruption."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.results = root / "synth_results.jsonl"
        self.batches = root / "synth_batches.json"

    def done_ids(self) -> set[str]:
        if not self.results.exists():
            return set()
        return {json.loads(line)["passage_id"] for line in self.results.open()}

    def append(self, passage_id: str, out: dict) -> None:
        with self.results.open("a") as f:
            f.write(json.dumps({"passage_id": passage_id, **out}) + "\n")

    def load(self) -> dict[str, dict]:
        if not self.results.exists():
            return {}
        return {(r := json.loads(line))["passage_id"]: r for line in self.results.open()}

    def pending_batches(self) -> list[str]:
        return json.loads(self.batches.read_text()) if self.batches.exists() else []

    def set_pending_batches(self, ids: list[str]) -> None:
        self.batches.write_text(json.dumps(ids))


def synth_sync(client: anthropic.Anthropic, model: str, passages: list[Passage], store: SynthStore) -> int:
    n = 0
    for p in passages:
        out = _parse(client.messages.create(**_params(model, p)))
        if out is not None:
            store.append(p.id, out)
            n += 1
    return n


def synth_batch(client: anthropic.Anthropic, model: str, passages: list[Passage], store: SynthStore,
                poll_s: int = 60, max_per_batch: int = 50_000) -> int:
    """Submit (or resume) batches, wait for them, and store results. Returns results stored."""
    pending = store.pending_batches()
    if not pending:
        done = store.done_ids()
        todo = [p for p in passages if p.id not in done]
        for i in range(0, len(todo), max_per_batch):
            part = todo[i : i + max_per_batch]
            batch = client.messages.batches.create(requests=[
                Request(custom_id=f"p{j}", params=MessageCreateParamsNonStreaming(**_params(model, p)))
                for j, p in enumerate(part, start=i)
            ])
            pending.append(batch.id)
            log.info("submitted batch %s (%d requests)", batch.id, len(part))
        store.set_pending_batches(pending)
        # custom_id -> passage id mapping (custom ids must be short)
        (store.results.parent / "synth_idmap.json").write_text(json.dumps({f"p{j}": p.id for j, p in enumerate(todo)}))

    idmap = json.loads((store.results.parent / "synth_idmap.json").read_text())
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
                store.append(idmap[result.custom_id], out)
                stored += 1
        pending.remove(batch_id)
        store.set_pending_batches(pending)
    return stored

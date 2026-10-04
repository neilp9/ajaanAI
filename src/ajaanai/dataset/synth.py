"""Question synthesis with Claude.

For each passage, Claude writes 1-3 questions a caller might genuinely ask that the passage
answers, and picks which of *his own sentences* (by number) make a short spoken answer.
Answers are therefore always his words — Claude never writes the teacher's side.

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

SYSTEM = """You help build a training set for a voice assistant that answers in the words of \
Thanissaro Bhikkhu, a Theravada monk. You will see one passage from his talks or writings, with \
each sentence numbered.

Write questions that a real person might ask him on the phone that this passage genuinely answers \
— practitioners, beginners, people going through something difficult. Vary the phrasing: casual, \
personal, specific. Don't mention "the passage" or "the talk" in the question.

For each question, choose the sentence numbers that, read in order, form a natural short spoken \
answer (2-6 sentences, ideally contiguous). Don't choose sentences that depend on missing context.

Mark the passage unusable if it is chanting, logistics, a fragment, mostly a quotation of someone \
else, or otherwise not a teaching that answers a question."""

SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "short_answer_sentences": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["question", "short_answer_sentences"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["usable", "topics", "items"],
    "additionalProperties": False,
}


def render(p: Passage) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(split_sentences(p.text)))
    return f"Title: {p.title}\n\n{numbered}\n\nWrite 1-3 questions."


def _params(model: str, p: Passage) -> dict:
    return dict(
        model=model,
        max_tokens=1024,
        system=SYSTEM,
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

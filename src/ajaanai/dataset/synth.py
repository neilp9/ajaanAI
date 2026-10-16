"""Conversation synthesis by an LLM (SYNTH_MODEL: anthropic:<model> or openai:<model>).

For each passage, the synth model scripts a short conversation — a voice call or a text chat, per
`passage_channel` — whose caller lines are its own (styled for that channel), and each of the
teacher's turns is a few of *his own sentences*, picked by number to answer the caller's line (about
a third of caller lines go off-script, so replies learn to respond rather than just continue the
passage). The only teacher-side words the model may write are an optional brief acknowledgement
before his sentences (`acknowledge`) and an optional short question back after them (`ask_back`);
both are kept apart in the results so the build can include or drop them (DATASET_ACKNOWLEDGE,
DATASET_ASK_BACK).

Real retreat Q&A exchanges (passage.kind == "qa") are handled the same way: the real question opens
the call and his real answer is spread over a few turns with brief caller follow-ups in between.

With DATASET_REPLY_MODE=shaped, the model instead *writes* each reply in his voice from the passage
(his wording, images and examples wherever they fit, lightly reshaped to answer the caller), and
lists the sentences it drew on (`sources`). See SYSTEM_SHAPED.

Bulk runs go through the provider's batch API (50% cheaper, resumable); `--sample` sends
concurrent requests for a quick look. `run_batch` / `run_sync` are shared with the responsiveness check
(check.py).

Stored results are matched by item id *and* a hash of the request's user message (`input_sha`), so an
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..llm import Backend, Reply, Request
from .chunk import Passage, passage_channel, split_sentences

log = logging.getLogger(__name__)

_INTRO = """You help build a training set for an assistant that talks with people — over voice calls \
and text chat — in the words of Thanissaro Bhikkhu, a Theravada monk. The conversations are real \
back-and-forth, not lectures: he makes one point at a time, briefly, and the caller responds."""

_CALLER_RULES = """Each exchange is the caller's line followed by his reply.

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
His replies are the same on both channels."""

_VERBATIM_RULES = """\
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

_RULES = f"{_CALLER_RULES}\n\n{_VERBATIM_RULES}"

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

# --- shaped replies (DATASET_REPLY_MODE=shaped) ----------------------------------------------------
# The synth model writes his replies from the passage instead of picking sentences, so a reply can meet what
# the caller actually said. Verbatim replies capped interactivity: judged by the eval's own judge,
# verbatim training conversations scored 2.25/5 ("recitation rather than conversation").

_SHAPED_RULES = """\
His replies (you write these, in his voice): each one answers what the caller just said, the way he \
would in a live conversation. The passage is your source, not a script: draw on its ideas, images, \
examples and phrasing, but use only what answers this caller's line, in whatever order serves the \
conversation, and leave the rest unused. Never walk through the passage in order or give its next \
sentence just because it comes next.
Keep his wording where it fits naturally (his turns of phrase and images are what make it sound like \
him), but rephrase freely so each reply is addressed to this person and what they said. Teach only \
what the passage says or clearly implies: no new doctrine, no claims, stories or quotations he doesn't \
give there.
Meet the person:
- When they share something personal (a relative, an illness, their work, a feeling), the reply takes \
it up first, specifically, before or while making his point. Never skip past it.
- When they doubt or push back ("that seems too simple", "I've tried everything"), answer the doubt \
itself: he'd challenge it, give them something concrete to try, or say why it's harder than it looks.
- When they answer his question, build on their answer.
- Engaging them never means going beyond the passage: don't label or diagnose their experience \
(naming a hindrance, a motive, a defilement) or stretch a teaching to their case unless the passage \
supports it. When it isn't clear what's going on with them, ask instead of deciding for them.
Hand it back: on about one reply in three, end with one short question (at most 12 words) that turns \
them toward their own experience, e.g. "What do you notice when that happens?", or end by giving \
them something specific to try. The caller's next line must respond to it.
His manner: plain, direct, warm but unsentimental, practical, with everyday analogies and now and \
then a dry touch of humor. One point at a time: usually 1-3 sentences, under about 60 words. Only \
when the caller asks him to say more or go deeper, set `longer` and use up to 6 sentences.
Every reply is heard as a direct answer, so don't open by continuing an earlier reply ("So", "And", \
"But", "Or", "Because", "This is why"). Never praise the question; no therapy-speak ("that sounds \
really hard", "that's completely normal"), no hedging, no lists or headings. Write for the ear: \
never mention "the passage", "the book", chapters, "tonight's talk" or earlier sessions, and leave \
out transcription glitches (repeated words, broken fragments).
`sources`: the numbers of the passage sentences the reply draws on (empty only for a reply that \
just takes up what the caller said)."""

SYSTEM_SHAPED = f"""{_INTRO}

You will see one passage from his talks or writings, with each sentence numbered. Write 1-2 \
conversations (2 if the passage covers two distinct points), each 3-6 exchanges, that this passage \
can carry. Callers are practitioners, beginners, or people going through something difficult.

{_CALLER_RULES}

{_SHAPED_RULES}

Mark the passage unusable if it is chanting, logistics, a fragment, mostly a quotation of someone \
else, or otherwise not a teaching that could answer a caller."""

SYSTEM_QA_SHAPED = f"""{_INTRO}

You will see a real question someone asked him at a retreat, and his real answer with each \
sentence numbered. Turn it into one conversation of 2-5 exchanges: the first caller line is \
the real question (copy it), and his answer is spread over the replies, with caller lines in between \
that the next part genuinely answers. Keep his answer roughly in order; you may leave out parts \
that don't fit a one-to-one conversation (references to earlier sessions, retreat logistics).

{_CALLER_RULES}

{_SHAPED_RULES}

Mark it unusable if the question is about logistics or the answer can't stand without the retreat \
context."""

SCHEMA_SHAPED = {
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
                                "reply": {"type": "string"},
                                "sources": {"type": "array", "items": {"type": "integer"}},
                                "longer": {"type": "boolean"},
                            },
                            "required": ["caller", "reply", "sources", "longer"],
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


def _params(model: str, p: Passage, shaped: bool = False) -> Request:
    qa = p.question is not None
    if shaped:
        # Written replies are longer than sentence numbers; room too for the model's own thinking.
        system, schema, max_tokens = (SYSTEM_QA_SHAPED if qa else SYSTEM_SHAPED), SCHEMA_SHAPED, 16000
    else:
        system, schema, max_tokens = (SYSTEM_QA if qa else SYSTEM), SCHEMA, 4096
    return Request(model=model, system=system, user=render(p), schema=schema, max_tokens=max_tokens)


def input_sha(req: Request) -> str:
    """Hash of what the model sees about the item (its message, not the model or system prompt).
    Hashed in the chat-message form it has always had, so results stored earlier stay current."""
    messages = [{"role": "user", "content": req.user}]
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


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


class _Usage:
    """Token totals for a run, logged at the end so a sample shows what a full run would cost."""

    def __init__(self):
        self.requests = self.input = self.output = 0

    def add(self, reply: Reply) -> None:
        self.requests += 1
        self.input += reply.input_tokens
        self.output += reply.output_tokens

    def log(self, what: str) -> None:
        if self.requests:
            log.info("%s: %d requests, %d input / %d output tokens (%.0f / %.0f per request)", what,
                     self.requests, self.input, self.output, self.input / self.requests,
                     self.output / self.requests)


def run_sync(backend: Backend, items: list[tuple[str, Request]], store: SynthStore, workers: int = 8) -> int:
    """Concurrent requests for (id, request) items not yet in the store. Returns results stored."""
    n, usage = 0, _Usage()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(backend.complete, req): (item_id, req) for item_id, req in store.todo(items)}
        for fut in as_completed(futures):
            item_id, req = futures[fut]
            reply = fut.result()
            usage.add(reply)
            if reply.data is not None:
                store.append(item_id, reply.data, input_sha(req))  # appended here, on one thread
                n += 1
    usage.log("sync")
    return n


def run_batch(backend: Backend, items: list[tuple[str, Request]], store: SynthStore,
              poll_s: int = 60, max_per_batch: int | None = None) -> int:
    """Submit (or resume) batches for (id, request) items not yet stored, wait, store results."""
    pending = store.pending_batches()
    if not pending:
        todo = store.todo(items)
        step = max_per_batch or backend.max_per_batch
        for k in range(0, len(todo), step):
            part = todo[k : k + step]
            batch_id = backend.submit([(f"p{j}", req) for j, (_, req) in enumerate(part, start=k)])
            pending.append(batch_id)
            log.info("submitted batch %s (%d requests)", batch_id, len(part))
        store.set_pending_batches(pending)
        # custom_id -> [item id, input hash] (custom ids must be short)
        store.idmap.write_text(json.dumps({f"p{j}": [i, input_sha(req)] for j, (i, req) in enumerate(todo)}))

    idmap = json.loads(store.idmap.read_text()) if store.idmap.exists() else {}
    stored, usage = 0, _Usage()
    for batch_id in list(pending):
        while not (st := backend.status(batch_id)).ended:
            log.info("batch %s: %d/%d done (%.0f%%; succeeded=%d errored=%d)", batch_id, st.done, st.total,
                     100 * st.done / max(st.total, 1), st.succeeded, st.errored)
            time.sleep(poll_s)
        for custom_id, reply in backend.results(batch_id):
            if reply is None:
                continue
            usage.add(reply)
            if reply.data is not None and custom_id in idmap:
                item_id, sha = idmap[custom_id]
                store.append(item_id, reply.data, sha)
                stored += 1
        pending.remove(batch_id)
        store.set_pending_batches(pending)
    usage.log("batch")
    return stored


def synth_sync(backend: Backend, model: str, passages: list[Passage], store: SynthStore) -> int:
    return run_sync(backend, [(p.id, _params(model, p)) for p in passages], store)


def synth_batch(backend: Backend, model: str, passages: list[Passage], store: SynthStore,
                poll_s: int = 60, max_per_batch: int | None = None) -> int:
    return run_batch(backend, [(p.id, _params(model, p)) for p in passages], store, poll_s, max_per_batch)

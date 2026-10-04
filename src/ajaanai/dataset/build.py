"""Assemble provider-neutral chat JSONL ({"messages": [...]}) from synthesized questions.

Example types (all assistant text is the teacher's own words):
  short     caller question -> 2-6 of his sentences          (phone register)
  grounded  same, with the source passage given as a reference (teaches using RAG context)
  long      caller question -> the full passage               (depth / longer explanations)
  talk      "give a short talk on X" -> opening of an evening talk

Split is by document, so no eval passage's talk leaks into training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .chunk import Passage, split_sentences, stable_bucket
from .persona import RAG_PREAMBLE, SYSTEM_PROMPT

MAX_SHORT_WORDS = 180


@dataclass
class BuildStats:
    train: int = 0
    eval: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    skipped_unusable: int = 0


def _msg(system: str, user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def _short_answer(sentences: list[str], ids: list[int]) -> str | None:
    picked = sorted({i for i in ids if 0 <= i < len(sentences)})
    if not 2 <= len(picked) <= 8:
        return None
    text = " ".join(sentences[i] for i in picked)
    return text if len(text.split()) <= MAX_SHORT_WORDS else None


def build_examples(passages: list[Passage], synth: dict[str, dict], eval_fraction: float,
                   talk_fraction: float = 0.1) -> tuple[list[dict], list[dict], list[dict], BuildStats]:
    """Returns (train, eval, eval_questions, stats)."""
    stats = BuildStats()
    train, evals, eval_qs = [], [], []
    seen_q: set[str] = set()
    talk_docs_done: set[str] = set()

    def add(example: dict, kind: str, is_eval: bool):
        (evals if is_eval else train).append(example)
        stats.by_type[kind] = stats.by_type.get(kind, 0) + 1

    for p in passages:
        is_eval = stable_bucket(p.doc_id) < eval_fraction * 10_000
        out = synth.get(p.id)
        if out is not None and not out.get("usable", False):
            stats.skipped_unusable += 1
        elif out is not None:
            sentences = split_sentences(p.text)
            for n, item in enumerate(out.get("items", [])):
                q = item["question"].strip()
                if not q or q.lower() in seen_q:
                    continue
                seen_q.add(q.lower())
                short = _short_answer(sentences, item.get("short_answer_sentences", []))
                if is_eval:
                    eval_qs.append({"question": q, "reference": p.text, "doc_id": p.doc_id,
                                    "passage_id": p.id, "title": p.title})
                if short:
                    grounded = stable_bucket(p.id + q) % 10 < 3
                    system = SYSTEM_PROMPT + ("\n\n" + RAG_PREAMBLE + p.text if grounded else "")
                    add(_msg(system, q, short), "grounded" if grounded else "short", is_eval)
                if n == 0:
                    add(_msg(SYSTEM_PROMPT, q, p.text), "long", is_eval)

        # A slice of evening talks as "give a talk" examples (first passage only).
        if (p.kind == "talk" and p.id.endswith("#0") and p.doc_id not in talk_docs_done
                and stable_bucket("talk:" + p.doc_id) < talk_fraction * 10_000):
            talk_docs_done.add(p.doc_id)
            add(_msg(SYSTEM_PROMPT, f"Could you give a short Dhamma talk on “{p.title}”?", p.text), "talk", is_eval)

    stats.train, stats.eval = len(train), len(evals)
    return train, evals, eval_qs, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

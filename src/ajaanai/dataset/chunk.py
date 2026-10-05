"""Split documents into passages on paragraph boundaries, and passages into sentences.

Retreat Q&A sections ("Q: …" / "A: …") are split into one passage per real exchange instead,
with the question kept alongside his answer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..catalog import Doc

_SENT = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"”’)]))\s+(?=[\"“‘(]?[A-Z0-9])")


@dataclass
class Passage:
    id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str
    author: str
    source: str
    kind: str
    text: str
    question: str | None = None  # set for a real Q&A exchange; `text` is then his answer

    @property
    def words(self) -> int:
        return len(self.text.split())


def split_sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        out += [s.strip() for s in _SENT.split(para.strip()) if s.strip()]
    return out


def chunk_text(text: str, min_words: int = 300, max_words: int = 700) -> list[str]:
    """Greedy paragraph packing. Oversized paragraphs are split on sentences."""
    paras: list[str] = []
    for p in (p.strip() for p in text.split("\n\n")):
        if not p:
            continue
        if len(p.split()) > max_words:
            buf, n = [], 0
            for s in split_sentences(p):
                buf.append(s)
                n += len(s.split())
                if n >= max_words // 2:
                    paras.append(" ".join(buf))
                    buf, n = [], 0
            if buf:
                paras.append(" ".join(buf))
        else:
            paras.append(p)

    chunks, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and n + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
        if n >= min_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
    if cur:
        tail = "\n\n".join(cur)
        if chunks and n < min_words // 2:
            chunks[-1] += "\n\n" + tail  # fold a short tail into the previous chunk
        else:
            chunks.append(tail)
    return chunks


def passages_for(doc: Doc, min_words: int = 300, max_words: int = 700) -> list[Passage]:
    return [
        Passage(id=f"{doc.id}#{i}", doc_id=doc.id, title=doc.title, author=doc.author,
                source=doc.source, kind=doc.kind, text=t)
        for i, t in enumerate(chunk_text(doc.text or "", min_words, max_words))
    ]


_QA_MARK = re.compile(r"(?m)^(Q|A):\s*")


def is_qa(doc: Doc) -> bool:
    return len(re.findall(r"(?m)^Q:\s", doc.text or "")) >= 2


def _strip_headers(answer: str) -> str:
    """Drop session headers ("February 14, 2026, afternoon", "Q&A") that trail an answer."""
    paras = [p.strip() for p in answer.split("\n\n") if p.strip()]
    while paras and (len(paras[-1].split()) < 6 and not paras[-1].endswith((".", "?", "!", "”"))):
        paras.pop()
    return "\n\n".join(paras)


def qa_passages(doc: Doc) -> list[Passage]:
    parts = _QA_MARK.split(doc.text or "")
    seq = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
    out = []
    for (k1, q), (k2, a) in zip(seq, seq[1:]):
        if k1 == "Q" and k2 == "A" and (a := _strip_headers(a)):
            out.append(Passage(id=f"{doc.id}#qa{len(out)}", doc_id=doc.id, title=doc.title, author=doc.author,
                               source=doc.source, kind="qa", text=a, question=" ".join(q.split())))
    return out


def stable_bucket(key: str, buckets: int = 10_000) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % buckets


TEXT_SHARE = 0.4


def passage_channel(p: Passage) -> str:
    """Which channel ("voice" | "text") this passage's conversations are written for. Fixed per
    passage: synthesis styles the caller's lines for it and the build uses the matching prompt,
    so changing TEXT_SHARE means re-synthesizing."""
    return "text" if stable_bucket("channel:" + p.id) < TEXT_SHARE * 10_000 else "voice"

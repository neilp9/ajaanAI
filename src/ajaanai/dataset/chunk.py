"""Split documents into passages on paragraph boundaries, and passages into sentences."""

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


def stable_bucket(key: str, buckets: int = 10_000) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % buckets

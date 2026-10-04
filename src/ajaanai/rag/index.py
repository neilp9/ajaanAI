"""Retrieval index over every passage with text (his own words *and* his translations).

Stored as plain files on the data volume:
  rag/passages.jsonl   one passage per line (id, doc_id, title, author, sha, text)
  rag/embeddings.npy   float16, L2-normalised, row i <-> line i
Incremental: new/changed docs are appended; superseded rows are dropped on the next rewrite.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..catalog import Catalog
from ..dataset.chunk import passages_for

log = logging.getLogger(__name__)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Embedder:
    """Thin wrapper so the embedding model is swappable (EMBED_MODEL)."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.has_query_prompt = "query" in (getattr(self.model, "prompts", None) or {})

    def docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        v = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float16)

    def query(self, text: str) -> np.ndarray:
        kw = {"prompt_name": "query"} if self.has_query_prompt else {}
        return np.asarray(self.model.encode([text], normalize_embeddings=True, **kw)[0], dtype=np.float32)


@dataclass
class Hit:
    score: float
    title: str
    author: str
    text: str
    doc_id: str


class RagIndex:
    def __init__(self, root: Path):
        self.root = root
        self.meta_path = root / "passages.jsonl"
        self.emb_path = root / "embeddings.npy"
        self.meta: list[dict] = []
        self.emb = np.zeros((0, 0), dtype=np.float16)
        if self.meta_path.exists() and self.emb_path.exists():
            self.meta = [json.loads(line) for line in self.meta_path.open()]
            self.emb = np.load(self.emb_path)

    def __len__(self) -> int:
        return len(self.meta)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.meta_path.open("w") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        np.save(self.emb_path, self.emb)

    def update(self, cat: Catalog, embedder: Embedder) -> int:
        """Embed passages for docs that are new or whose text changed. Returns rows added."""
        have = {}
        for m in self.meta:
            have.setdefault(m["doc_id"], m["doc_sha"])
        todo, changed_docs = [], set()
        for doc in cat.iter_with_text():
            sha = _sha(doc.text)
            if have.get(doc.id) == sha:
                continue
            if doc.id in have:
                changed_docs.add(doc.id)
            for p in passages_for(doc, min_words=150, max_words=350):
                todo.append({"id": p.id, "doc_id": doc.id, "doc_sha": sha, "title": doc.title,
                             "author": doc.author, "text": p.text})
        if changed_docs:
            keep = [i for i, m in enumerate(self.meta) if m["doc_id"] not in changed_docs]
            self.meta = [self.meta[i] for i in keep]
            self.emb = self.emb[keep] if len(self.emb) else self.emb
        if not todo:
            return 0
        log.info("embedding %d passages", len(todo))
        vecs = embedder.docs([t["text"] for t in todo])
        self.emb = vecs if self.emb.size == 0 else np.vstack([self.emb, vecs])
        self.meta += todo
        self.save()
        return len(todo)

    def search(self, qvec: np.ndarray, k: int) -> list[Hit]:
        if not self.meta:
            return []
        scores = self.emb.astype(np.float32) @ qvec
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(float(scores[i]), self.meta[i]["title"], self.meta[i]["author"], self.meta[i]["text"],
                    self.meta[i]["doc_id"]) for i in top]


class Retriever:
    def __init__(self, index: RagIndex, embedder: Embedder, k: int):
        self.index, self.embedder, self.k = index, embedder, k

    def __call__(self, query: str) -> list[Hit]:
        return self.index.search(self.embedder.query(query), self.k)


def format_hits(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        who = "" if h.author == "thanissaro" else " (his translation of another teacher)"
        parts.append(f"[{h.title}{who}]\n{h.text}")
    return "\n\n---\n\n".join(parts)

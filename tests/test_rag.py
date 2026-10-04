import zlib

import numpy as np

from ajaanai.catalog import Catalog, Doc
from ajaanai.rag.index import RagIndex, format_hits


class FakeEmbedder:
    """Bag-of-words hashing embedder: similar texts -> similar vectors."""

    def _vec(self, text):
        v = np.zeros(1024, dtype=np.float32)
        for w in text.lower().split():
            v[zlib.crc32(w.strip(".,").encode()) % 1024] += 1
        return v / (np.linalg.norm(v) or 1)

    def docs(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float16)

    def query(self, text):
        return self._vec(text)


def add(cat, i, text, author="thanissaro"):
    cat.upsert_listing(Doc(id=i, source="evening", kind="talk", title=i, page_url="/" + i, author=author))
    cat.set_text(i, text, "site")


def test_incremental_update_and_search(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite")
    add(cat, "breath", " ".join(["The breath is your home base in meditation."] * 40))
    add(cat, "anger", " ".join(["When anger arises, look at it as a disease of the mind."] * 40), "translation")
    idx, emb = RagIndex(tmp_path / "rag"), FakeEmbedder()
    assert idx.update(cat, emb) > 0
    assert idx.update(cat, emb) == 0  # nothing changed

    hits = RagIndex(tmp_path / "rag").search(emb.query("what should I do when anger arises"), 1)
    assert hits[0].doc_id == "anger"
    assert "his translation of another teacher" in format_hits(hits)

    # A doc's text changes (ASR -> official transcript): its old rows are replaced, not duplicated.
    before = len(idx)
    cat.upsert_listing(Doc(id="breath", source="evening", kind="talk", title="breath", page_url="/breath"))
    cat.db.execute("UPDATE docs SET text_source='asr:x' WHERE id='breath'")
    cat.set_text("breath", " ".join(["Stay with the breath, gently and steadily."] * 40), "site")
    idx.update(cat, emb)
    assert len(idx) <= before
    assert all("home base" not in m["text"] for m in idx.meta if m["doc_id"] == "breath")

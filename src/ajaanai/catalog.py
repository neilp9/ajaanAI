"""SQLite catalog of every document we know about (talks, book chapters, essays).

One row per document. Scrapers upsert listings; transcript text arrives either from the
site (`text_source='site'`) or from ASR (`text_source='asr:<provider>'`). An official site
transcript always wins over ASR text, never the other way round.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id            TEXT PRIMARY KEY,          -- stable id derived from the page path
    source        TEXT NOT NULL,             -- 'evening' or a sources.yaml key
    kind          TEXT NOT NULL,             -- talk | book_section | article
    author        TEXT NOT NULL DEFAULT 'thanissaro',  -- thanissaro | translation | other
    title         TEXT NOT NULL,
    date          TEXT,                      -- ISO date the talk was given (talks only)
    page_url      TEXT NOT NULL,
    mp3_url       TEXT,
    text          TEXT,
    text_source   TEXT,                      -- site | asr:<provider>
    status        TEXT NOT NULL,             -- has_text | needs_transcript | failed
    sha256        TEXT,
    meta          TEXT NOT NULL DEFAULT '{}',
    first_seen    TEXT NOT NULL,
    last_checked  TEXT,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS docs_status ON docs(status, source);
"""

HAS_TEXT = "has_text"
NEEDS_TRANSCRIPT = "needs_transcript"
FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Doc:
    id: str
    source: str
    kind: str
    title: str
    page_url: str
    author: str = "thanissaro"
    date: str | None = None
    mp3_url: str | None = None
    text: str | None = None
    text_source: str | None = None
    status: str = NEEDS_TRANSCRIPT
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Doc":
        return cls(
            id=row["id"], source=row["source"], kind=row["kind"], title=row["title"],
            page_url=row["page_url"], author=row["author"], date=row["date"],
            mp3_url=row["mp3_url"], text=row["text"], text_source=row["text_source"],
            status=row["status"], meta=json.loads(row["meta"] or "{}"),
        )


class Catalog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        with self.db:
            yield self.db

    def close(self) -> None:
        self.db.close()

    # --- writes ----------------------------------------------------------------

    def upsert_listing(self, doc: Doc) -> bool:
        """Insert a newly discovered doc, or refresh its listing fields.

        Never clobbers existing text. Returns True if the doc is new.
        """
        now = _now()
        existing = self.db.execute("SELECT id FROM docs WHERE id=?", (doc.id,)).fetchone()
        with self.tx():
            if existing:
                self.db.execute(
                    """UPDATE docs SET title=?, date=COALESCE(?, date), page_url=?,
                       mp3_url=COALESCE(?, mp3_url), author=?, kind=?, updated_at=? WHERE id=?""",
                    (doc.title, doc.date, doc.page_url, doc.mp3_url, doc.author, doc.kind, now, doc.id),
                )
            else:
                self.db.execute(
                    """INSERT INTO docs (id, source, kind, author, title, date, page_url, mp3_url,
                       status, meta, first_seen, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (doc.id, doc.source, doc.kind, doc.author, doc.title, doc.date, doc.page_url,
                     doc.mp3_url, NEEDS_TRANSCRIPT, json.dumps(doc.meta), now, now),
                )
        if doc.text:
            self.set_text(doc.id, doc.text, doc.text_source or "site")
        return existing is None

    def set_text(self, doc_id: str, text: str, text_source: str) -> bool:
        """Store transcript text. Site text replaces ASR text; ASR never replaces site text.

        Returns True if the stored text changed.
        """
        row = self.db.execute("SELECT text_source, sha256 FROM docs WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(doc_id)
        if row["text_source"] == "site" and text_source != "site":
            return False
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest == row["sha256"]:
            return False
        now = _now()
        with self.tx():
            self.db.execute(
                """UPDATE docs SET text=?, text_source=?, sha256=?, status=?, last_checked=?,
                   updated_at=? WHERE id=?""",
                (text, text_source, digest, HAS_TEXT, now, now, doc_id),
            )
        return True

    def mark_checked(self, doc_id: str) -> None:
        with self.tx():
            self.db.execute("UPDATE docs SET last_checked=? WHERE id=?", (_now(), doc_id))

    def mark_failed(self, doc_id: str, error: str) -> None:
        row = self.get(doc_id)
        meta = {**(row.meta if row else {}), "error": error[:500]}
        with self.tx():
            self.db.execute(
                "UPDATE docs SET status=?, meta=?, updated_at=? WHERE id=?",
                (FAILED, json.dumps(meta), _now(), doc_id),
            )

    # --- reads -------------------------------------------------------------------

    def get(self, doc_id: str) -> Doc | None:
        row = self.db.execute("SELECT * FROM docs WHERE id=?", (doc_id,)).fetchone()
        return Doc.from_row(row) if row else None

    def find_by_mp3(self, mp3_url: str) -> Doc | None:
        row = self.db.execute("SELECT * FROM docs WHERE mp3_url=? LIMIT 1", (mp3_url,)).fetchone()
        return Doc.from_row(row) if row else None

    def exists(self, doc_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM docs WHERE id=?", (doc_id,)).fetchone() is not None

    def pending_recheck(self, source: str, recheck_days: int, today: date | None = None) -> list[Doc]:
        """Docs without a site transcript, young enough that one may still be posted.

        Undated docs (e.g. 2002 talks named '0211n5-...') age from when we first catalogued them,
        so they're re-checked for `recheck_days` and then dropped rather than forever.
        """
        cutoff = ((today or date.today()) - timedelta(days=recheck_days)).isoformat()
        rows = self.db.execute(
            """SELECT * FROM docs WHERE source=? AND (text_source IS NULL OR text_source != 'site')
               AND COALESCE(date, substr(first_seen, 1, 10)) >= ? ORDER BY date DESC""",
            (source, cutoff),
        ).fetchall()
        return [Doc.from_row(r) for r in rows]

    def ready_for_asr(
        self, grace_days: int, sources: list[str] | None = None, limit: int | None = None,
        today: date | None = None, include_failed: bool = False,
    ) -> list[Doc]:
        """Docs with audio but no text, older than the grace period."""
        cutoff = ((today or date.today()) - timedelta(days=grace_days)).isoformat()
        statuses = [NEEDS_TRANSCRIPT] + ([FAILED] if include_failed else [])
        q = f"""SELECT * FROM docs WHERE status IN ({",".join("?" * len(statuses))})
                AND mp3_url IS NOT NULL AND (date IS NULL OR date <= ?)"""
        args: list = [*statuses, cutoff]
        if sources:
            q += f" AND source IN ({','.join('?' * len(sources))})"
            args += sources
        q += " ORDER BY date DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [Doc.from_row(r) for r in self.db.execute(q, args).fetchall()]

    def iter_with_text(self, authors: list[str] | None = None, sources: list[str] | None = None) -> Iterator[Doc]:
        q, args = "SELECT * FROM docs WHERE status=?", [HAS_TEXT]
        if authors:
            q += f" AND author IN ({','.join('?' * len(authors))})"
            args += authors
        if sources:
            q += f" AND source IN ({','.join('?' * len(sources))})"
            args += sources
        for row in self.db.execute(q + " ORDER BY id", args):
            yield Doc.from_row(row)

    def counts(self) -> list[dict]:
        rows = self.db.execute(
            """SELECT source, status, COALESCE(text_source,'-') AS text_source, COUNT(*) AS n
               FROM docs GROUP BY source, status, text_source ORDER BY source, status"""
        ).fetchall()
        return [dict(r) for r in rows]


def open_catalog(path: Path | None = None) -> Catalog:
    from .config import get_settings

    return Catalog(path or get_settings().catalog_path)

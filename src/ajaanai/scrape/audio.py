"""Shared ingestion for audio talk listings (used by the evening component and audio sources)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from ..catalog import Catalog, Doc
from .http import Disallowed, PoliteClient
from .parsers.common import doc_id_from_path
from .parsers.talks import TalkListing, parse_talk_page

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    seen: int = 0
    new: int = 0
    site_transcripts: int = 0  # transcripts fetched from the site (new or upgraded from ASR)
    duplicates: int = 0  # same mp3 already catalogued under another source
    errors: list[str] = field(default_factory=list)

    def merge(self, other: "IngestStats") -> "IngestStats":
        self.seen += other.seen
        self.new += other.new
        self.site_transcripts += other.site_transcripts
        self.duplicates += other.duplicates
        self.errors += other.errors
        return self


def ingest_listings(
    cat: Catalog, http: PoliteClient, listings: list[TalkListing], *, source: str,
    author: str = "thanissaro", fetch_pages: str = "new", limit: int | None = None,
    checkpoint: Callable[[], None] | None = None, checkpoint_every: int = 50,
) -> IngestStats:
    """Upsert listings; fetch talk pages for transcripts.

    fetch_pages: "new" (only pages for docs we haven't stored text for), "all", or "none".
    """
    stats = IngestStats()
    for listing in (listings if limit is None else listings[:limit]):
        stats.seen += 1
        if checkpoint and stats.seen % checkpoint_every == 0:
            checkpoint()
        doc_id = doc_id_from_path(listing.page_url)
        if not listing.has_page and listing.mp3_url:
            dup = cat.find_by_mp3(listing.mp3_url)
            if dup is not None and dup.id != doc_id:
                stats.duplicates += 1
                continue
        doc = Doc(
            id=doc_id, source=source, kind="talk", author=author, title=listing.title,
            page_url=listing.page_url, mp3_url=listing.mp3_url,
            date=listing.date.isoformat() if listing.date else None,
        )
        if cat.upsert_listing(doc):
            stats.new += 1
        if not listing.has_page or fetch_pages == "none":
            continue
        current = cat.get(doc_id)
        if fetch_pages == "new" and current and current.text_source == "site":
            continue
        try:
            page = parse_talk_page(http.text(listing.page_url))
        except Disallowed:
            continue
        except Exception as e:  # keep going; one bad page shouldn't stop a crawl
            stats.errors.append(f"{listing.page_url}: {e}")
            log.warning("failed to fetch %s: %s", listing.page_url, e)
            continue
        if page.text and cat.set_text(doc_id, page.text, "site"):
            stats.site_transcripts += 1
        else:
            cat.mark_checked(doc_id)
    return stats

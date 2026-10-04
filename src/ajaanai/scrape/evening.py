"""Component 1: evening talks, run on a schedule.

Each run:
  1. discovers new talks (RSS + current-year index; or every year page with --backfill);
  2. re-checks recent talks that still lack an official transcript, and swaps one in when it
     appears (official text replaces any ASR text);
  3. transcribes talks still missing text after the grace period (configurable provider).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from ..catalog import Catalog
from ..config import Settings
from .audio import IngestStats, ingest_listings
from .http import PoliteClient
from .parsers.talks import parse_index, parse_rss, parse_talk_page, parse_year_links

log = logging.getLogger(__name__)

SOURCE = "evening"
CURRENT_INDEX = "/mp3_index.html"
RSS = "/rss/evening.xml"
PREFIX = "/audio/evening/"


@dataclass
class EveningReport:
    discovered: IngestStats = field(default_factory=IngestStats)
    rechecked: int = 0
    upgraded_to_site: int = 0
    transcribed: int = 0
    transcribe_failed: int = 0
    changed_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        d = self.discovered
        return (f"evening: seen={d.seen} new={d.new} site_transcripts={d.site_transcripts} "
                f"rechecked={self.rechecked} upgraded={self.upgraded_to_site} "
                f"asr_ok={self.transcribed} asr_failed={self.transcribe_failed} errors={len(d.errors)}")


def parse_years(spec: str | None) -> set[int] | None:
    """'2000-2005,2010' -> {2000..2005, 2010}."""
    if not spec:
        return None
    years: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        years.update(range(int(a), int(b or a) + 1))
    return years


def discover(http: PoliteClient, backfill: bool, years: set[int] | None):
    current_html = http.text(CURRENT_INDEX)
    listings = parse_index(current_html)
    if backfill:
        for link in parse_year_links(current_html, PREFIX):
            year = int(link.strip("/").rsplit("/", 1)[-1])
            if years and year not in years:
                continue
            listings += parse_index(http.text(link))
        if years:
            listings = [x for x in listings if x.date and x.date.year in years]
    else:
        listings = parse_rss(http.text(RSS)) + listings
    seen, unique = set(), []
    for x in listings:
        if x.page_url not in seen:
            seen.add(x.page_url)
            unique.append(x)
    return unique


def run_evening(
    cat: Catalog, http: PoliteClient, settings: Settings, *, backfill: bool = False,
    years: str | None = None, limit: int | None = None, transcribe: bool = True,
    today: date | None = None, checkpoint: Callable[[], None] | None = None,
) -> EveningReport:
    """`checkpoint` is called periodically so long runs persist progress (Modal: volume.commit)."""
    report = EveningReport()
    listings = discover(http, backfill, parse_years(years))
    log.info("evening: %d listings discovered", len(listings))
    report.discovered = ingest_listings(cat, http, listings, source=SOURCE, fetch_pages="new", limit=limit,
                                        checkpoint=checkpoint)
    handled = {x.page_url for x in (listings if limit is None else listings[:limit])}

    # Re-check recent talks still lacking an official transcript.
    for doc in cat.pending_recheck(SOURCE, settings.recheck_days, today=today):
        if doc.page_url in handled:
            continue
        report.rechecked += 1
        if checkpoint and report.rechecked % 50 == 0:
            checkpoint()
        try:
            page = parse_talk_page(http.text(doc.page_url))
        except Exception as e:
            log.warning("recheck failed for %s: %s", doc.id, e)
            continue
        if page.text and cat.set_text(doc.id, page.text, "site"):
            report.upgraded_to_site += 1
            report.changed_ids.append(doc.id)
        else:
            cat.mark_checked(doc.id)

    if transcribe:
        from ..transcribe import transcribe_pending

        ok, failed = transcribe_pending(cat, settings, sources=[SOURCE], today=today, checkpoint=checkpoint)
        report.transcribed, report.transcribe_failed = len(ok), len(failed)
        report.changed_ids += ok

    log.info(report.summary())
    return report

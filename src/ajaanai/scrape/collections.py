"""Component 2: on-demand, config-driven collection of everything else (books, essays, other
audio series). Sources are declared in sources.yaml; run them by name with `ajaanai collect`.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ..catalog import Catalog, Doc
from .audio import IngestStats, ingest_listings
from .http import PoliteClient
from .parsers.books import book_dir, parse_book_toc, parse_books_index, parse_section
from .parsers.common import doc_id_from_path, parse_html
from .parsers.talks import parse_index, parse_year_links

log = logging.getLogger(__name__)


class SourceSpec(BaseModel):
    kind: Literal["book", "article", "audio_series"]
    parser: Literal["book", "audio_index", "article_index"]
    start_urls: list[str]
    description: str = ""
    include: list[str] = Field(default_factory=list)  # globs matched against url (and book key)
    exclude: list[str] = Field(default_factory=list)
    author: Literal["auto", "thanissaro", "translation", "other"] = "auto"
    follow_years: bool = False  # audio_index: also crawl /YYYY/ archive pages
    skip_anthologies: bool = False  # book: skip study guides / canon anthologies
    transcribe: bool = False  # audio: send untranscribed items to ASR
    enabled: bool = False
    refresh: str | None = None  # optional cron; picked up by the Modal scheduler


class SourcesFile(BaseModel):
    sources: dict[str, SourceSpec]


def load_sources(path: Path) -> dict[str, SourceSpec]:
    return SourcesFile.model_validate(yaml.safe_load(path.read_text())).sources


def _match(spec: SourceSpec, *candidates: str) -> bool:
    def any_match(patterns):
        return any(fnmatch.fnmatch(c, p) for p in patterns for c in candidates if c)

    if spec.include and not any_match(spec.include):
        return False
    return not any_match(spec.exclude)


@dataclass
class CollectResult:
    name: str
    planned: list[str] = field(default_factory=list)  # urls (dry run) or doc ids
    stats: IngestStats = field(default_factory=IngestStats)


def _author(spec: SourceSpec, detected: str = "thanissaro") -> str:
    return detected if spec.author == "auto" else spec.author


# --- per-parser collectors -------------------------------------------------------


def _collect_audio(name, spec, cat, http, dry_run, limit, checkpoint=None) -> CollectResult:
    res = CollectResult(name)
    listings = []
    for start in spec.start_urls:
        html = http.text(start)
        listings += parse_index(html)
        if spec.follow_years:
            for link in parse_year_links(html, start if start.startswith("/audio/") else "/audio/x/"):
                listings += parse_index(http.text(link))
    listings = [x for x in listings if _match(spec, x.page_url, x.mp3_url or "")]
    if dry_run:
        res.planned = [x.page_url for x in (listings if limit is None else listings[:limit])]
        return res
    res.stats = ingest_listings(cat, http, listings, source=name, author=_author(spec), limit=limit,
                                checkpoint=checkpoint)
    return res


def _collect_books(name, spec, cat, http, dry_run, limit, checkpoint=None) -> CollectResult:
    res = CollectResult(name)
    books = []
    for start in spec.start_urls:
        books += parse_books_index(http.text(start))
    books = [b for b in books if _match(spec, b.url, b.key) and not (spec.skip_anthologies and b.anthology)]
    for book in (books if limit is None else books[:limit]):
        author = _author(spec, book.author)
        root = book_dir(book.url)
        sections = parse_book_toc(http.text(root), root) or [book.url]
        if dry_run:
            res.planned += [f"{book.key} [{author}{', anthology' if book.anthology else ''}] {root} ({len(sections)} sections)"]
            continue
        if checkpoint:
            checkpoint()
        for i, path in enumerate(sections):
            doc_id = doc_id_from_path(path)
            res.stats.seen += 1
            existing = cat.get(doc_id)
            if existing and existing.text:
                continue
            try:
                title, text = parse_section(http.text(path))
            except Exception as e:
                res.stats.errors.append(f"{path}: {e}")
                continue
            if not text:
                continue
            doc = Doc(id=doc_id, source=name, kind="book_section", author=author,
                      title=f"{book.title} — {title}" if title else book.title, page_url=path,
                      text=text, text_source="site",
                      meta={"book": book.key, "book_title": book.title, "section": i,
                            "anthology": book.anthology})
            if cat.upsert_listing(doc):
                res.stats.new += 1
                res.stats.site_transcripts += 1
    return res


def _collect_articles(name, spec, cat, http, dry_run, limit, checkpoint=None) -> CollectResult:
    res = CollectResult(name)
    urls: list[str] = []
    for start in spec.start_urls:
        for a in parse_html(http.text(start)).css("a[href]"):
            href = a.attributes["href"].split("#", 1)[0]
            if href.endswith(".html") and href.startswith("/") and _match(spec, href) and href not in urls:
                urls.append(href)
    urls = urls if limit is None else urls[:limit]
    if dry_run:
        res.planned = urls
        return res
    for path in urls:
        doc_id = doc_id_from_path(path)
        res.stats.seen += 1
        if checkpoint and res.stats.seen % 50 == 0:
            checkpoint()
        if (d := cat.get(doc_id)) and d.text:
            continue
        try:
            title, text = parse_section(http.text(path))
        except Exception as e:
            res.stats.errors.append(f"{path}: {e}")
            continue
        if text:
            cat.upsert_listing(Doc(id=doc_id, source=name, kind="article", author=_author(spec),
                                   title=title or path, page_url=path, text=text, text_source="site"))
            res.stats.new += 1
    return res


def _cron_field_matches(field: str, value: int) -> bool:
    for part in field.split(","):
        rng, _, step = part.partition("/")
        if rng == "*":
            lo, hi = 0, 10_000
        elif "-" in rng:
            lo, hi = (int(x) for x in rng.split("-"))
        else:
            lo = hi = int(rng)
        if lo <= value <= hi and (value - (lo if rng != "*" else 0)) % int(step or 1) == 0:
            return True
    return False


def cron_due_this_hour(expr: str, now) -> bool:
    """Standard 5-field cron, evaluated at hour granularity (the minute field is ignored)."""
    _, hour, dom, month, dow = expr.split()
    return (_cron_field_matches(hour, now.hour) and _cron_field_matches(dom, now.day)
            and _cron_field_matches(month, now.month)
            and _cron_field_matches(dow, (now.weekday() + 1) % 7))  # cron: 0 = Sunday


def due_sources(sources: dict[str, SourceSpec], now) -> list[str]:
    return [n for n, s in sources.items() if s.refresh and cron_due_this_hour(s.refresh, now)]


COLLECTORS = {"audio_index": _collect_audio, "book": _collect_books, "article_index": _collect_articles}


def collect(
    sources: dict[str, SourceSpec], names: list[str] | None, cat: Catalog, http: PoliteClient,
    *, dry_run: bool = False, limit: int | None = None, transcribe: bool = True,
    checkpoint=None,
) -> list[CollectResult]:
    if names:
        unknown = set(names) - set(sources)
        if unknown:
            raise SystemExit(f"Unknown source(s): {', '.join(sorted(unknown))}. See sources.yaml")
        chosen = {n: sources[n] for n in names}  # explicit names run even if disabled
    else:
        chosen = {n: s for n, s in sources.items() if s.enabled}
    results = []
    for name, spec in chosen.items():
        log.info("collecting %s (%s)", name, spec.parser)
        res = COLLECTORS[spec.parser](name, spec, cat, http, dry_run, limit, checkpoint)
        results.append(res)
        if not dry_run and transcribe and spec.transcribe:
            from ..config import get_settings
            from ..transcribe import transcribe_pending

            ok, failed = transcribe_pending(cat, get_settings(), sources=[name], grace_days=0,
                                            checkpoint=checkpoint)
            log.info("%s: transcribed %d, failed %d", name, len(ok), len(failed))
    return results

"""Parsers for the /books/ catalogue, a book's table of contents, and a book section page."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .common import block_text, clean, parse_html

# Authors other than Thanissaro Bhikkhu whose works he translated (or that appear on /books/).
_OTHER_AUTHOR = re.compile(
    r"\b(?:by|of)\s+(?:Phra\s+|Venerable\s+)?(?:Ajaan Lee|Ajaan Suwat|Phra Ajaan|Ajahn Chah|"
    r"Ācariya Mahā Boowa|Venerable Ācariya|The Dhammayut Order)",
    re.IGNORECASE,
)
_ANTHOLOGY = re.compile(r"\b(study guide|anthology|compiled by|translation, side-by-side)\b", re.IGNORECASE)


@dataclass
class BookListing:
    key: str  # the <li id>, e.g. 'Undaunted'
    title: str
    url: str  # '/books/Undaunted/' (or a direct Section page)
    blurb: str
    author: str  # thanissaro | translation
    anthology: bool  # mostly canon passages (study guides / anthologies)


def parse_books_index(html: str) -> list[BookListing]:
    tree = parse_html(html)
    out: list[BookListing] = []
    seen: set[str] = set()
    for li in tree.css("li.list-group-item[id]"):
        a = li.css_first("a.title") or li.css_first("a[href]")
        if a is None:
            continue
        href = a.attributes.get("href") or ""
        if not href.startswith("/books/") or href.rstrip("/") == "/books":
            continue
        key = li.attributes["id"]
        if key in seen:
            continue
        seen.add(key)
        blurb = clean(li.text(separator=" "))
        out.append(BookListing(
            key=key,
            title=clean(a.text()),
            url=href,
            blurb=blurb,
            author="translation" if _OTHER_AUTHOR.search(blurb) else "thanissaro",
            anthology=bool(_ANTHOLOGY.search(blurb)),
        ))
    return out


def book_dir(url: str) -> str:
    """'/books/Undaunted/Section0001.html' or '/books/Undaunted' -> '/books/Undaunted/'."""
    m = re.match(r"(/books/[^/#?]+)", url)
    return (m.group(1) if m else url.rstrip("/")) + "/"


def parse_book_toc(html: str, book_url: str) -> list[str]:
    """Ordered, de-duplicated section page paths for a book."""
    base = book_dir(book_url)
    tree = parse_html(html)
    out: list[str] = []
    for a in tree.css("a[href]"):
        href = a.attributes["href"].split("#", 1)[0]
        if href.startswith(base) and re.search(r"/(Section\d+|Chapter\d+|ch\d+)[^/]*\.html$", href, re.I):
            if href not in out:
                out.append(href)
    return out


def parse_section(html: str) -> tuple[str, str]:
    """(title, text) of a book section page."""
    tree = parse_html(html)
    main = tree.css_first("main.prose") or tree.css_first("div.prose") or tree.body
    heading = main.css_first("h1, h2, h3") if main is not None else None
    title = clean(heading.text()) if heading else ""
    return title, block_text(main) if main is not None else ""

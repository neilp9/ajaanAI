"""Parsers for audio talk index pages, talk pages and the RSS feed.

Layout (2026 site):
  index:  <li class="li-audio"> <button class="play" data-link="/Archive/y2024/241006_X.mp3">
          <span class="smalldate">06</span> <a class="audio" href="/audio/evening/2024/241006-x.html">
          <span>Title</span></a></li>
  talk:   <h1>Title</h1> ... <audio src="/Archive/...mp3"> ... <div class="prose">transcript</div>
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from urllib.parse import unquote

from .common import block_text, clean, date_from_slug, parse_html

NOT_TRANSCRIBED_MARKERS = ("hasn't been transcribed", "has not been transcribed")


@dataclass
class TalkListing:
    title: str
    page_url: str  # site-relative path (equals mp3_url when there is no talk page)
    mp3_url: str | None
    date: date | None

    @property
    def has_page(self) -> bool:
        return not self.page_url.lower().endswith(".mp3")


@dataclass
class TalkPage:
    title: str
    mp3_url: str | None
    text: str | None  # None when the site has no transcript yet


def parse_index(html: str) -> list[TalkListing]:
    """Talk listings on an audio index page.

    Handles both layouts: `li.li-audio` rows (evening/morning/lectures) and bare `a.audio` links
    that point straight at an mp3 (topical / collections / guided meditations).
    """
    tree = parse_html(html)
    out: list[TalkListing] = []
    rows = tree.css("li.li-audio") or tree.css("a.audio")
    for row in rows:
        a = row if row.tag == "a" else row.css_first("a.audio")
        if a is None or not a.attributes.get("href"):
            continue
        href = unquote(a.attributes["href"])
        play = row.css_first("button.play") if row.tag != "a" else None
        mp3 = play.attributes.get("data-link") if play else None
        if not mp3:
            dl = href if href.lower().endswith(".mp3") else None
            if dl is None and row.tag != "a":
                node = row.css_first('a[href$=".mp3"]')
                dl = node.attributes.get("href") if node else None
            mp3 = dl
        mp3 = unquote(mp3) if mp3 else None
        title_node = a.css_first("span:not(.smalldate)") or a
        title = clean(title_node.text())
        if not title and mp3:
            title = clean(re.sub(r"[_]+", " ", mp3.rsplit("/", 1)[-1].rsplit(".", 1)[0]))
        out.append(TalkListing(
            title=title,
            page_url=href,
            mp3_url=mp3,
            date=date_from_slug(href.rsplit("/", 1)[-1]) or (date_from_slug(mp3.rsplit("/", 1)[-1]) if mp3 else None),
        ))
    return out


def parse_year_links(html: str, prefix: str = "/audio/evening/") -> list[str]:
    """Year archive links on an index page, e.g. ['/audio/evening/2000/', ...]."""
    tree = parse_html(html)
    prefix = "/" + prefix.strip("/") + "/"
    years = set()
    for a in tree.css("a[href]"):
        href = a.attributes["href"].replace("https://www.dhammatalks.org", "")
        rest = href[len(prefix):].strip("/") if href.startswith(prefix) else ""
        if rest.isdigit() and len(rest) == 4:
            years.add(f"{prefix}{rest}/")
    return sorted(years)


def parse_talk_page(html: str) -> TalkPage:
    tree = parse_html(html)
    h1 = tree.css_first("h1")
    audio = tree.css_first("audio[src]") or tree.css_first('a[href$=".mp3"]')
    mp3 = None
    if audio is not None:
        mp3 = audio.attributes.get("src") or audio.attributes.get("href")
    prose = tree.css_first("div.prose") or tree.css_first("main.prose")
    text = block_text(prose) if prose is not None else ""
    if not text or any(m in text.lower() for m in NOT_TRANSCRIBED_MARKERS):
        text = None
    return TalkPage(title=clean(h1.text()) if h1 else "", mp3_url=mp3, text=text)


def parse_rss(xml_text: str) -> list[TalkListing]:
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        enc = item.find("enclosure")
        mp3 = enc.get("url") if enc is not None else None
        path = link.replace("https://www.dhammatalks.org", "")
        out.append(TalkListing(
            title=clean(item.findtext("title") or ""),
            page_url=path,
            mp3_url=mp3.replace("https://www.dhammatalks.org", "") if mp3 else None,
            date=date_from_slug(path.rsplit("/", 1)[-1]),
        ))
    return out

from __future__ import annotations

import re
from datetime import date

from selectolax.lexbor import LexborHTMLParser, LexborNode

BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "pre"}
_WS = re.compile(r"\s+")
# '241006-guardians-at-death' / '241006_Guardians.mp3' / '031220 Limitless.mp3' / '260930(short)_X.mp3'
_YYMMDD = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?=[-_. (])")
# '20140426-Thanissaro_Bhikkhu-IMC-...mp3'
_YYYYMMDD = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")


def parse_html(html: str) -> LexborHTMLParser:
    return LexborHTMLParser(html)


def clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def block_text(root: LexborNode) -> str:
    """Readable text of a content container: one paragraph per block, blank-line separated.

    Only outermost block elements are used so nested blocks (p inside li) aren't duplicated.
    """
    paras: list[str] = []
    for node in root.css(",".join(sorted(BLOCK_TAGS))):
        parent, nested = node.parent, False
        while parent is not None and parent is not root and parent.tag != "html":
            if parent.tag in BLOCK_TAGS:
                nested = True
                break
            parent = parent.parent
        if nested:
            continue
        t = clean(node.text(separator=" "))
        if t:
            paras.append(t)
    return "\n\n".join(paras)


def date_from_slug(slug: str) -> date | None:
    """Talk dates are encoded as yymmdd (or yyyymmdd) in page slugs and mp3 names (all 2000+)."""
    if m := _YYYYMMDD.search(slug):
        y, mm, dd = (int(g) for g in m.groups())
    elif m := _YYMMDD.search(slug):
        y, mm, dd = (int(g) for g in m.groups())
        y += 2000
    else:
        return None
    try:
        return date(y, mm, dd)
    except ValueError:
        return None


def doc_id_from_path(path: str) -> str:
    """'/audio/evening/2024/241006-guardians-at-death.html' -> 'audio/evening/2024/241006-guardians-at-death'."""
    path = path.split("#", 1)[0].split("?", 1)[0]
    path = re.sub(r"^https?://[^/]+", "", path).strip("/")
    return re.sub(r"\.html?$", "", path)

from datetime import date
from pathlib import Path

from ajaanai.catalog import HAS_TEXT, NEEDS_TRANSCRIPT, Catalog, Doc
from ajaanai.config import Settings
from ajaanai.scrape.evening import parse_years, run_evening

F = Path(__file__).parent / "fixtures"


def make_doc(i="audio/evening/2026/x", d="2026-09-01"):
    return Doc(id=i, source="evening", kind="talk", title="X", page_url=f"/{i}.html",
               mp3_url="/a.mp3", date=d)


def test_site_text_beats_asr(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite")
    cat.upsert_listing(make_doc())
    assert cat.get("audio/evening/2026/x").status == NEEDS_TRANSCRIPT
    assert cat.set_text("audio/evening/2026/x", "asr words", "asr:elevenlabs")
    assert cat.set_text("audio/evening/2026/x", "official words", "site")  # upgrade
    assert not cat.set_text("audio/evening/2026/x", "asr again", "asr:openai")  # never downgrade
    d = cat.get("audio/evening/2026/x")
    assert (d.text, d.text_source, d.status) == ("official words", "site", HAS_TEXT)


def test_upsert_is_idempotent_and_keeps_text(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite")
    assert cat.upsert_listing(make_doc()) is True
    cat.set_text("audio/evening/2026/x", "words", "site")
    assert cat.upsert_listing(make_doc()) is False
    assert cat.get("audio/evening/2026/x").text == "words"


def test_grace_period_and_recheck_windows(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite")
    cat.upsert_listing(make_doc("a/new", "2026-09-30"))
    cat.upsert_listing(make_doc("a/old", "2026-08-01"))
    cat.upsert_listing(make_doc("a/ancient", "2020-01-01"))
    today = date(2026, 10, 4)
    assert [d.id for d in cat.ready_for_asr(21, today=today)] == ["a/old", "a/ancient"]
    assert {d.id for d in cat.pending_recheck("evening", 120, today=today)} == {"a/new", "a/old"}


def test_undated_docs_age_out_of_recheck(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite")
    cat.upsert_listing(make_doc("a/undated", None))
    seen = cat.get("a/undated")
    assert seen.date is None
    first_seen = date.fromisoformat(cat.db.execute(
        "SELECT substr(first_seen, 1, 10) FROM docs WHERE id='a/undated'").fetchone()[0])
    from datetime import timedelta

    assert [d.id for d in cat.pending_recheck("evening", 120, today=first_seen)] == ["a/undated"]
    assert [d.id for d in cat.pending_recheck("evening", 120, today=first_seen + timedelta(days=119))] == ["a/undated"]
    assert cat.pending_recheck("evening", 120, today=first_seen + timedelta(days=121)) == []


def test_parse_years():
    assert parse_years("2000-2002,2010") == {2000, 2001, 2002, 2010}
    assert parse_years(None) is None


class FakeHttp:
    """Serves fixtures by path, records requests."""

    def __init__(self, pages):
        self.pages, self.requested = pages, []

    def text(self, path):
        self.requested.append(path)
        return self.pages[path]


def test_evening_run_discovers_and_upgrades(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite")
    talk_html = (F / "talk_transcribed.html").read_text()
    untranscribed = (F / "talk_untranscribed.html").read_text()
    pages = {
        "/mp3_index.html": (F / "current_index.html").read_text(),
        "/rss/evening.xml": (F / "evening_rss.xml").read_text(),
    }
    http = FakeHttp(pages)

    class Pages(dict):
        def __missing__(self, key):
            return untranscribed

    http.pages = Pages(pages)
    s = Settings(_env_file=None, data_dir=tmp_path)
    r = run_evening(cat, http, s, limit=5, transcribe=False, today=date(2026, 10, 4))
    assert r.discovered.new == 5 and r.discovered.site_transcripts == 0

    # A transcript appears later for one of them -> the daily run picks it up.
    target = cat.pending_recheck("evening", 120, today=date(2026, 10, 4))[0]
    http.pages[target.page_url] = talk_html
    r2 = run_evening(cat, http, s, limit=0, transcribe=False, today=date(2026, 10, 4))
    assert r2.discovered.seen == 0  # limit=0 means "no discovery", so the recheck pass must find it
    assert r2.upgraded_to_site == 1
    assert cat.get(target.id).text_source == "site"


def test_source_refresh_cron():
    from datetime import datetime

    from ajaanai.scrape.collections import SourceSpec, cron_due_this_hour, due_sources

    mon_10 = datetime(2026, 10, 5, 10, 5)  # a Monday
    assert cron_due_this_hour("0 10 * * 1", mon_10)
    assert not cron_due_this_hour("0 10 * * 2", mon_10)
    assert cron_due_this_hour("30 */2 * * *", mon_10)
    assert not cron_due_this_hour("0 9-11/3 1 * *", mon_10)
    spec = dict(kind="book", parser="book", start_urls=["/books/"])
    srcs = {"a": SourceSpec(**spec, refresh="0 10 * * 1"), "b": SourceSpec(**spec)}
    assert due_sources(srcs, mon_10) == ["a"]

from datetime import date
from pathlib import Path

from ajaanai.scrape.parsers import books, talks
from ajaanai.scrape.parsers.common import date_from_slug, doc_id_from_path

F = Path(__file__).parent / "fixtures"


def read(name):
    return (F / name).read_text()


def test_current_index_rows():
    rows = talks.parse_index(read("current_index.html"))
    assert len(rows) == 226
    first = rows[0]
    assert first.title == "Cornered Through Samvega"
    assert first.page_url == "/audio/evening/2026/261003-cornered-through-samvega.html"
    assert first.mp3_url == "/Archive/y2026/261003_Cornered_Through_Samvega.mp3"
    assert first.date == date(2026, 10, 3)
    assert all(r.has_page and r.date for r in rows)


def test_year_links_cover_all_years():
    links = talks.parse_year_links(read("current_index.html"), "/audio/evening/")
    assert links[0] == "/audio/evening/2000/"
    assert "/audio/evening/2025/" in links


def test_old_year_index_unquotes_mp3_paths():
    rows = talks.parse_index(read("year_2005.html"))
    assert len(rows) == 201
    assert rows[0].mp3_url == "/Archive/y2005/051231 Proving the Teachings.mp3"


def test_mp3_only_index_layouts():
    lectures = talks.parse_index(read("lectures_index.html"))
    assert lectures and not any(r.has_page for r in lectures)
    topical = talks.parse_index(read("topical_index.html"))
    assert topical[0].mp3_url == "/Archive/y2003/031220 Limitless Thoughts.mp3"
    assert topical[0].date == date(2003, 12, 20)


def test_talk_page_transcribed_vs_not():
    t = talks.parse_talk_page(read("talk_transcribed.html"))
    assert t.title == "Guardians at Death"
    assert t.text.startswith("When we meditate, we take our breath as home base.")
    assert "\n\n" in t.text
    u = talks.parse_talk_page(read("talk_untranscribed.html"))
    assert u.text is None
    assert u.mp3_url.endswith("When_a_Loved_One_Has_Died.mp3")


def test_rss():
    items = talks.parse_rss(read("evening_rss.xml"))
    assert len(items) == 50
    assert items[0].page_url.startswith("/audio/evening/2026/")


def test_books_author_detection():
    bs = {b.key: b for b in books.parse_books_index(read("books.html"))}
    assert bs["Undaunted"].author == "thanissaro"
    assert bs["heightenedMind"].author == "translation"
    assert bs["craftoftheheart"].author == "translation"
    assert bs["wings"].anthology is True


def test_book_toc_and_section():
    toc = books.parse_book_toc(read("book_index.html"), "/books/Undaunted/")
    assert toc[0] == "/books/Undaunted/Section0001.html"
    assert len(toc) == len(set(toc))
    title, text = books.parse_section(read("book_section.html"))
    assert title == "Introduction" and len(text) > 1000


def test_date_and_id_helpers():
    assert date_from_slug("20140426-Thanissaro_Bhikkhu-IMC-x.mp3") == date(2014, 4, 26)
    assert date_from_slug("260930(short)_Medicine.mp3") == date(2026, 9, 30)
    assert date_from_slug("0208n3a1 Seeker's Habits.mp3") is None
    assert doc_id_from_path("https://www.dhammatalks.org/audio/evening/2024/x.html#a") == "audio/evening/2024/x"

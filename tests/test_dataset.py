from ajaanai.catalog import Doc
from ajaanai.dataset.build import build_examples
from ajaanai.dataset.chunk import chunk_text, passages_for, split_sentences
from ajaanai.dataset.persona import SYSTEM_PROMPT


def para(n, word="breath"):
    return " ".join([f"The {word} is home."] * n)


def test_chunker_respects_bounds():
    text = "\n\n".join(para(20) for _ in range(20))  # 20 paras x 80 words
    chunks = chunk_text(text, 300, 700)
    sizes = [len(c.split()) for c in chunks]
    assert all(s <= 700 for s in sizes)
    assert all(s >= 150 for s in sizes)
    assert sum(sizes) == len(text.split())


def test_split_sentences():
    assert split_sentences("One. Two? “Three!” Four") == ["One.", "Two?", "“Three!”", "Four"]


def test_build_examples_uses_only_his_sentences():
    doc = Doc(id="d1", source="evening", kind="talk", title="T", page_url="/x",
              text="\n\n".join(" ".join(f"Sentence {p}-{i} here." for i in range(10)) for p in range(8)))
    ps = passages_for(doc)
    synth = {ps[0].id: {"usable": True, "topics": [], "items": [
        {"question": "How do I start?", "short_answer_sentences": [0, 1, 2]},
        {"question": "how do i start?", "short_answer_sentences": [3, 4]},  # duplicate (case) dropped
    ]}}
    train, evals, eval_qs, stats = build_examples(ps, synth, eval_fraction=0.0, talk_fraction=1.0)
    rows = train + evals
    short = [r for r in rows if r["messages"][1]["content"] == "How do I start?"
             and len(r["messages"][2]["content"].split()) < 20]
    assert short and short[0]["messages"][2]["content"] == "Sentence 0-0 here. Sentence 0-1 here. Sentence 0-2 here."
    assert all(r["messages"][0]["content"].startswith(SYSTEM_PROMPT) for r in rows)
    assert stats.by_type.get("talk") == 1
    assert sum(1 for r in rows if r["messages"][1]["content"].lower() == "how do i start?") == 2  # short/grounded + long


def test_unusable_passages_skipped():
    doc = Doc(id="d2", source="evening", kind="talk", title="T", page_url="/x", text=para(100))
    ps = passages_for(doc)
    train, evals, _, stats = build_examples(ps, {ps[0].id: {"usable": False, "topics": [], "items": []}}, 0.0, 0.0)
    assert stats.skipped_unusable == 1 and not train

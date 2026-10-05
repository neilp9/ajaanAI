from ajaanai.catalog import Doc
from ajaanai.dataset.build import build_examples, conversation_turns
from ajaanai.dataset.chunk import chunk_text, is_qa, passage_channel, passages_for, qa_passages, split_sentences
from ajaanai.dataset.persona import system_prompt


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


def talk_passage():
    doc = Doc(id="d1", source="evening", kind="talk", title="T", page_url="/x",
              text="\n\n".join(" ".join(f"Sentence {p}-{i} here." for i in range(10)) for p in range(8)))
    return passages_for(doc)


def ex(caller, sentences, ask_back="", longer=False):
    return {"caller": caller, "sentences": sentences, "longer": longer, "ask_back": ask_back}


def test_conversation_is_multi_turn_and_only_his_sentences():
    ps = talk_passage()
    synth = {ps[0].id: {"usable": True, "topics": [], "conversations": [{"exchanges": [
        ex("How do I start?", [0, 1]),
        ex("Then what?", [2], ask_back="What do you notice?"),
        ex("Tension, mostly.", [3, 4]),
    ]}, {"exchanges": [ex("how do i start?", [5])]}]}}  # duplicate opening (case) dropped
    train, evals, eval_qs, stats = build_examples(ps, synth, eval_fraction=0.0)
    assert len(train) == 1 and not evals
    msgs = train[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(system_prompt(passage_channel(ps[0])))
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * 3
    assert msgs[2]["content"] == "Sentence 0-0 here. Sentence 0-1 here."
    assert msgs[4]["content"] == "Sentence 0-2 here. What do you notice?"
    assert stats.assistant_turns == 3 and stats.ask_backs == 1


def test_bad_turn_ends_the_conversation_before_it():
    ps = talk_passage()
    s = split_sentences(ps[0].text)
    convo = {"exchanges": [ex("Q1", [0]), ex("Q2", [0, 1]), ex("Q3", [2])]}  # turn 2 reuses sentence 0
    turns, _, dropped = conversation_turns(ps[0], convo)
    assert [m["content"] for m in turns] == ["Q1", s[0]] and dropped == 2
    too_long = {"exchanges": [ex("Q1", [0, 1, 2, 3, 4])]}  # 5 sentences without `longer`
    assert conversation_turns(ps[0], too_long)[0] == []
    assert conversation_turns(ps[0], {"exchanges": [ex("Q1", [0, 1, 2, 3, 4], longer=True)]})[0]


def test_first_reply_must_stand_alone_but_later_ones_may_lean_back():
    doc = Doc(id="d3", source="evening", kind="talk", title="T", page_url="/x",
              text="So that's why it matters. The breath is home. Because it's always here. “Evil is done by oneself.”")
    p = passages_for(doc)[0]
    assert conversation_turns(p, {"exchanges": [ex("Q", [0])]})[0] == []
    assert conversation_turns(p, {"exchanges": [ex("Q", [2])]})[0] == []
    assert conversation_turns(p, {"exchanges": [ex("Q", [3])]})[0] == []  # a quotation, not his words
    turns, _, _ = conversation_turns(p, {"exchanges": [ex("Q", [1]), ex("Why?", [2])]})
    assert turns[-1]["content"] == "Because it's always here."


def test_ask_back_can_be_left_out_without_breaking_the_call():
    ps = talk_passage()
    convo = {"exchanges": [ex("Q1", [0], ask_back="What do you notice?"), ex("Tension.", [1])]}
    with_q, n, _ = conversation_turns(ps[0], convo)
    assert len(with_q) == 4 and n == 1
    without, n, dropped = conversation_turns(ps[0], convo, ask_back=False)
    assert len(without) == 2 and n == 0 and dropped == 1 and "?" not in without[1]["content"]
    rambling = {"exchanges": [ex("Q1", [0], ask_back="And " * 20 + "you?"), ex("Q2", [1])]}
    assert len(conversation_turns(ps[0], rambling)[0]) == 2  # over-long ask-back: call ends there


def test_qa_passages_keep_the_real_question():
    text = ("Q&A\n\nQ: What is karma?\n\nA: Karma is intentional action. It shapes what comes.\n\n"
            "February 15, 2026, afternoon\n\nQ&A\n\nQ: Why practice?\n\nA: To end suffering. That's all.")
    doc = Doc(id="b1", source="books", kind="book_section", title="K — Q&A", page_url="/x", text=text)
    assert is_qa(doc)
    ps = qa_passages(doc)
    assert [(p.question, p.text) for p in ps] == [
        ("What is karma?", "Karma is intentional action. It shapes what comes."),
        ("Why practice?", "To end suffering. That's all."),
    ]
    synth = {ps[0].id: {"usable": True, "topics": [], "conversations": [{"exchanges": [
        ex("paraphrased question", [0]), ex("Go on.", [1])]}]}}
    train, _, _, stats = build_examples(ps, synth, 0.0)
    assert train[0]["messages"][1]["content"] == "What is karma?"
    assert set(stats.by_type) <= {"qa", "qa_grounded"}


def test_unusable_passages_skipped():
    doc = Doc(id="d2", source="evening", kind="talk", title="T", page_url="/x", text=para(100))
    ps = passages_for(doc)
    train, evals, _, stats = build_examples(ps, {ps[0].id: {"usable": False, "topics": [], "conversations": []}}, 0.0)
    assert stats.skipped_unusable == 1 and not train


def test_finetune_expands_one_example_per_teacher_turn():
    from ajaanai.train.finetune import to_prompt_completions

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"}]
    out = to_prompt_completions({"messages": [msgs]})
    assert out["completion"] == [[msgs[2]], [msgs[4]]]
    assert out["prompt"] == [msgs[:2], msgs[:4]]


def test_channels_split_and_prompted_per_passage():
    doc = Doc(id="d4", source="evening", kind="talk", title="T", page_url="/x",
              text="\n\n".join(" ".join(f"Sentence {p}-{i} here." for i in range(60)) for p in range(200)))
    ps = passages_for(doc)
    channels = [passage_channel(p) for p in ps]
    assert channels == [passage_channel(p) for p in ps]  # stable
    assert 0.2 < channels.count("text") / len(ps) < 0.6
    synth = {p.id: {"usable": True, "topics": [], "conversations": [{"exchanges": [ex(f"Q {p.id}", [0])]}]} for p in ps}
    train, _, _, stats = build_examples(ps, synth, 0.0)
    for p, row in zip(ps, train):
        assert row["messages"][0]["content"].startswith(system_prompt(passage_channel(p)))
    assert stats.by_channel == {"text": channels.count("text"), "voice": channels.count("voice")}
    assert "text chat" in system_prompt("text") and "phone call" in system_prompt("voice")

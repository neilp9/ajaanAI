from ajaanai.train.eval import Arm, converse, reply_stats, summarize


def fake_ask(replies):
    it = iter(replies)
    calls = []

    def ask(arm, messages, channel):
        calls.append((arm.model_for(channel), [m["role"] for m in messages]))
        return next(it)

    return ask, calls


def test_conversation_alternates_and_feeds_the_arm_the_whole_history():
    ask, calls = fake_ask(["Watch the breath. What do you notice?", "Stay with that.", "Good."])
    lines = iter(["Tension in my chest.", "ok, then what?"])
    q = {"question": "How do I start?", "reference": "r", "channel": "text"}
    msgs = converse(Arm("a", "http://x/v1"), q, turns=3, next_line=lambda m, ch: next(lines), ask_fn=ask)
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3
    assert msgs[2]["content"] == "Tension in my chest."
    assert calls[0] == ("ajaan-text", ["user"]) and calls[2][1] == ["user", "assistant"] * 2 + ["user"]


def test_caller_can_end_early_and_explicit_models_are_kept():
    ask, calls = fake_ask(["A.", "B."])
    q = {"question": "Q", "reference": "r"}  # no channel recorded: voice
    msgs = converse(Arm("raw", "http://x/v1", "Qwen/Qwen3.5-27B"), q, turns=4, next_line=lambda m, ch: None, ask_fn=ask)
    assert len(msgs) == 2 and calls == [("Qwen/Qwen3.5-27B", ["user"])]
    assert Arm("p", "http://x/v1").model_for("voice") == "ajaan-voice"


def test_reply_stats_and_summary():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "One two three. Do you see?"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "Here's how:\n- first\n- second"}]
    stats = reply_stats(msgs)
    assert stats == {"replies": 2, "reply_words": [6, 6], "asks_back": 1, "markdown": 1}
    rows = [{"arm": "a", **stats, "style": 4, "fidelity": 5, "brevity": 3, "interactivity": 4},
            {"arm": "a", "error": "boom"}]
    s = summarize(rows, "a")
    assert s["n"] == 1 and s["errors"] == 1 and s["brevity"] == 3
    assert s["asks_back_share"] == 0.5 and s["markdown_share"] == 0.5 and s["reply_words_median"] == 6

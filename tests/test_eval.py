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


def test_each_run_gets_its_own_files(tmp_path):
    from ajaanai.train.eval import results_stem, write_results

    assert results_stem([Arm("tuned", "u"), Arm("base model", "u")], when="20261006-1142") == "20261006-1142-tuned-base_model"
    path = write_results(tmp_path, [Arm("tuned", "u")], [{"arm": "tuned"}], {"tuned": {"n": 1}})
    assert path.exists() and path.with_name(path.name.replace(".rows.jsonl", ".summary.json")).exists()


def test_markdown_catches_italics_but_not_plain_text():
    from ajaanai.train.eval import _MARKDOWN

    assert _MARKDOWN.search("the self *exists* only as a strategy")
    assert not _MARKDOWN.search("Breathe in. 3 x 4 = 12, and that's fine. Use a * if you must")


GRADES = {"style": 4, "fidelity": 5, "brevity": 3, "interactivity": 2, "note": "ok"}


class RecordingBackend:
    def __init__(self, data):
        self.data, self.requests = data, []

    def complete(self, req):
        from ajaanai.llm import Reply

        self.requests.append(req)
        return Reply(self.data)


def test_judge_and_caller_go_through_any_backend():
    from ajaanai.llm import ModelSpec
    from ajaanai.train.eval import JUDGE_SYSTEM, SCHEMA, caller_line, grader

    backend = RecordingBackend(GRADES)
    g = grader(ModelSpec("openai", "m"), backend)
    assert g({"reference": "ref", "channel": "text"}, [{"role": "user", "content": "hi"}]) == GRADES
    (req,) = backend.requests
    assert req.model == "m" and req.system == JUDGE_SYSTEM and req.schema == SCHEMA
    assert "Channel: text chat" in req.user and req.effort == "medium"

    caller = RecordingBackend({"line": " And then? ", "end": False})
    assert caller_line(caller, "c", [{"role": "assistant", "content": "Watch."}], "voice") == "And then?"
    assert "voice call" in caller.requests[0].system
    assert caller_line(RecordingBackend({"line": "", "end": True}), "c", [], "voice") is None
    assert caller_line(RecordingBackend(None), "c", [], "voice") is None


def test_several_judges_agreement_and_rejudge():
    from ajaanai.train.eval import agreement, rejudge, score_all, summaries

    graders = {"a": lambda q, m: GRADES, "b": lambda q, m: {**GRADES, "interactivity": 4}, "c": lambda q, m: None}
    row = {"arm": "x", **score_all(graders, {"reference": "r"}, [])}
    assert row["interactivity"] == 2 and row["judges"]["b"]["interactivity"] == 4  # first judge is the headline
    assert row["judges"]["c"] == {"error": "judge declined"}
    agree = agreement([row, row, row], "a", "b")
    assert agree["n"] == 3 and agree["interactivity"]["mean_diff"] == 2 and agree["style"]["exact"] == 1

    # an older row: flat scores, no `judges`; its reference comes from the questions file
    old = {"arm": "x", "question": "Q?", "channel": "voice", "conversation": [], **GRADES}
    (new,) = rejudge([old], {"b": graders["b"]}, {"Q?": {"reference": "r"}})
    assert set(new["judges"]) == {"original", "b"} and new["judges"]["original"]["interactivity"] == 2
    s = summaries([new], ["x"], ["original", "b"])
    assert s["x"]["by_judge"]["b"]["interactivity"] == 4 and s["_agreement"]["b vs original"]["n"] == 1

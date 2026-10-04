import asyncio
import json
import random

import httpx
import pytest

from ajaanai.models import get_profile
from ajaanai.serve.proxy import (Delta, ProxyConfig, ThinkStripper, build_upstream_body, create_app,
                                 should_think, spoken_stream)

QWEN = get_profile("qwen3.5-27b")
LLAMA = get_profile("llama-3.1-8b")


def cfg(**kw):
    base = dict(upstream_base_url="http://up/v1", upstream_model="m", upstream_api_key=None, profile=QWEN,
                filler_interval_s=0.05)
    return ProxyConfig(**{**base, **kw})


async def gen(items, delay=0.0):
    for it in items:
        await asyncio.sleep(delay)
        yield it


async def collect(agen):
    return [d async for d in agen]


def text_of(deltas):
    return "".join(d.content for d in deltas)


# --- reasoning decision ------------------------------------------------------------------------


@pytest.mark.parametrize("q,expected", [
    ("hello", False), ("thanks so much", False), ("can you hear me?", False),
    ("Why does the Buddha say clinging is suffering?", True),
    ("How should I deal with anger when meditating at night?", True),
    ("What time is it", False),
])
def test_should_think_auto(q, expected):
    assert should_think("auto", q) is expected


def test_should_think_modes():
    assert should_think("on", "hi") and not should_think("off", "Why is there suffering in this world?")


# --- think stripping -------------------------------------------------------------------------


def feed_all(stripper, chunks):
    return "".join(stripper.feed(c) for c in chunks) + stripper.flush()


def test_strips_inline_think_split_across_chunks():
    s = ThinkStripper()
    assert feed_all(s, ["Hel", "lo <thi", "nk>secret plan</th", "ink> world"]) == "Hello  world"


def test_implicit_open_tag_when_expecting_reasoning():
    s = ThinkStripper(expect_reasoning=True)
    assert feed_all(s, ["reasoning about ", "it\n</think>\n\n", "The answer."]) == "\n\nThe answer."


def test_no_reasoning_flushes_after_holdback():
    s = ThinkStripper(expect_reasoning=True, holdback_chars=10)
    assert feed_all(s, ["Just a plain answer with no tags."]) == "Just a plain answer with no tags."


# --- filler streaming --------------------------------------------------------------------------


async def test_no_fillers_when_not_thinking():
    out = await collect(spoken_stream(gen([Delta(content="Hi there."), Delta(finish_reason="stop")]), cfg(), think=False))
    assert text_of(out) == "Hi there."
    assert out[-1].finish_reason == "stop"


async def test_first_filler_is_immediate_and_reasoning_is_hidden():
    up = [Delta(reasoning="let me think"), Delta(content="The breath "), Delta(content="is home."),
          Delta(finish_reason="stop")]
    out = await collect(spoken_stream(gen(up), cfg(), think=True, rng=random.Random(1)))
    assert out[0].content.endswith("… ")  # filler first
    assert "let me think" not in text_of(out)
    assert text_of(out).endswith("The breath is home.")


async def test_repeat_fillers_during_long_silence_then_stop_once_answering():
    async def slow():
        yield Delta(reasoning="hmm")
        await asyncio.sleep(0.18)  # ~3 filler intervals of silence
        yield Delta(content="Answer.")
        await asyncio.sleep(0.18)  # silence after answering started: no fillers
        yield Delta(finish_reason="stop")

    out = await collect(spoken_stream(slow(), cfg(), think=True, rng=random.Random(2)))
    contents = [d.content for d in out if d.content]
    answer_at = contents.index("Answer.")
    assert answer_at >= 3  # first filler + at least two repeats
    assert contents[answer_at + 1:] == []
    assert all(a != b for a, b in zip(contents, contents[1:]))  # never the same filler twice in a row


async def test_filler_mode_off_and_always():
    up = [Delta(content="Yes."), Delta(finish_reason="stop")]
    off = await collect(spoken_stream(gen(up), cfg(filler_mode="off"), think=True))
    assert text_of(off) == "Yes."
    always = await collect(spoken_stream(gen(up), cfg(filler_mode="always", profile=LLAMA), think=False))
    assert text_of(always) != "Yes." and text_of(always).endswith("Yes.")


# --- request building -----------------------------------------------------------------------


def test_upstream_body_model_specific_kwargs():
    body = {"messages": [{"role": "system", "content": "EL agent prompt"}, {"role": "user", "content": "q"}]}
    q = build_upstream_body(cfg(), body, think=False, references="REF")
    assert q["chat_template_kwargs"] == {"enable_thinking": False}
    assert "REF" in q["messages"][0]["content"] and "EL agent prompt" in q["messages"][0]["content"]
    assert [m["role"] for m in q["messages"]] == ["system", "user"]
    l = build_upstream_body(cfg(profile=LLAMA), body, think=False, references=None)
    assert "chat_template_kwargs" not in l
    hosted = build_upstream_body(cfg(send_template_kwargs=False), body, think=True, references=None)
    assert "chat_template_kwargs" not in hosted


# --- end-to-end SSE through the FastAPI app with a mocked upstream ------------------------------


def upstream_sse(chunks):
    lines = []
    for c in chunks:
        lines.append("data: " + json.dumps({"choices": [{"index": 0, "delta": c, "finish_reason": None}]}))
    lines.append("data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


async def test_app_streams_openai_compatible_sse(respx_mock):
    respx_mock.post("http://up/v1/chat/completions").mock(return_value=httpx.Response(
        200, text=upstream_sse([{"reasoning_content": "x"}, {"content": "Be patient."}]),
        headers={"content-type": "text/event-stream"}))
    app = create_app(cfg(bearer_secret="s3cret", reasoning_mode="on"), retrieve=lambda q: "REFERENCE")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        r = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401
        r = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer s3cret"},
                              json={"stream": True, "messages": [{"role": "user", "content": "How do I meditate?"}]})
    events = [l[6:] for l in r.text.split("\n\n") if l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    spoken = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert spoken.endswith("Be patient.") and spoken != "Be patient."  # filler came first
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    sent = json.loads(respx_mock.calls.last.request.content)
    assert "REFERENCE" in sent["messages"][0]["content"]
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}

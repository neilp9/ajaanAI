import json
from types import SimpleNamespace as NS

import httpx
import pytest

from ajaanai.llm import AnthropicBackend, ModelSpec, OpenAIBackend, Request, make_backend, parse_spec, parse_specs

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


def req(model="m", **kw):
    return Request(model=model, system="sys", user="hello", schema=SCHEMA, max_tokens=100, **kw)


def test_specs_always_name_the_provider():
    assert parse_spec(" openai:gpt-x ") == ModelSpec("openai", "gpt-x")
    assert parse_spec("anthropic:claude-x").name == "anthropic:claude-x"
    assert [s.name for s in parse_specs("anthropic:a, openai:b")] == ["anthropic:a", "openai:b"]
    assert [s.name for s in parse_specs(["anthropic:a", "openai:b,openai:c"])] == ["anthropic:a", "openai:b", "openai:c"]
    for bad in ("claude-opus-5-5", "gpt-5", "gemini:x", "openai:", "openai: "):
        with pytest.raises(ValueError, match="anthropic:<model> or openai:<model>"):
            parse_spec(bad)


def test_make_backend_picks_by_provider_and_needs_an_openai_key():
    s = NS(anthropic_api_key="k", openai_api_key=None, openai_base_url="https://api.openai.com/v1")
    assert isinstance(make_backend(parse_spec("anthropic:m"), s), AnthropicBackend)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        make_backend(parse_spec("openai:m"), s)
    local = NS(anthropic_api_key=None, openai_api_key=None, openai_base_url="http://localhost:8000/v1")
    assert isinstance(make_backend(parse_spec("openai:m"), local), OpenAIBackend)  # local servers need no key


# --- Anthropic -----------------------------------------------------------------------------------


def a_message(payload, stop="end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return NS(stop_reason=stop, content=[NS(type="text", text=text)], usage=NS(input_tokens=7, output_tokens=3))


class FakeAnthropic:
    def __init__(self, *messages):
        self.queue, self.calls = list(messages), []
        self.messages = NS(create=self._create, batches=NS(create=self._batch, retrieve=self._retrieve,
                                                           results=self._results))
        self.beta = NS(messages=NS(create=lambda **kw: self._create(beta=True, **kw)))
        self.status = "in_progress"

    def _create(self, beta=False, **kw):
        self.calls.append((beta, kw))
        return self.queue.pop(0)

    def _batch(self, requests):
        self.calls.append(("batch", requests))
        return NS(id="msgbatch_1")

    def _retrieve(self, bid):
        return NS(processing_status=self.status, request_counts=NS(processing=1, succeeded=2, errored=1, canceled=0,
                                                                   expired=0))

    def _results(self, bid):
        yield NS(custom_id="p0", result=NS(type="succeeded", message=a_message({"ok": True})))
        yield NS(custom_id="p1", result=NS(type="errored", message=None))


def test_anthropic_complete():
    client = FakeAnthropic(a_message({"ok": True}), a_message({"ok": True}, "max_tokens"), a_message("not json"))
    b = AnthropicBackend(client=client)
    r = b.complete(req())
    assert r.data == {"ok": True} and (r.input_tokens, r.output_tokens) == (7, 3)
    beta, kw = client.calls[0]
    assert not beta and kw["system"] == "sys" and kw["messages"] == [{"role": "user", "content": "hello"}]
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert b.complete(req()).data is None  # cut off
    assert b.complete(req()).data is None  # unparseable


def test_anthropic_effort_and_fallback_only_on_models_that_take_them():
    client = FakeAnthropic(a_message({"ok": True}), a_message({"ok": True}))
    b = AnthropicBackend(client=client)
    b.complete(req("claude-opus-5-5", effort="medium", fallback=True))
    b.complete(req("claude-haiku-4-5", effort="medium", fallback=True))
    (beta1, kw1), (beta2, kw2) = client.calls
    assert beta1 and kw1["output_config"]["effort"] == "medium" and kw1["fallbacks"] == "default"
    assert not beta2 and "effort" not in kw2["output_config"] and "fallbacks" not in kw2


def test_anthropic_batch():
    client = FakeAnthropic()
    b = AnthropicBackend(client=client)
    assert b.submit([("p0", req()), ("p1", req())]) == "msgbatch_1"
    requests = client.calls[0][1]
    assert [r["custom_id"] for r in requests] == ["p0", "p1"] and requests[0]["params"]["model"] == "m"
    st = b.status("msgbatch_1")
    assert not st.ended and (st.done, st.total, st.succeeded, st.errored) == (3, 4, 2, 1)
    client.status = "ended"
    assert b.status("msgbatch_1").ended
    (c0, r0), (c1, r1) = b.results("msgbatch_1")
    assert (c0, r0.data, c1, r1) == ("p0", {"ok": True}, "p1", None)


# --- OpenAI-compatible ---------------------------------------------------------------------------


def completion(content, finish="stop", refusal=None):
    return {"choices": [{"finish_reason": finish, "message": {"content": content, "refusal": refusal}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def openai_backend(handler):
    calls = []

    def record(request):
        calls.append(request)
        return handler(request)

    b = OpenAIBackend("https://api.example/v1/", "k", http=httpx.Client(transport=httpx.MockTransport(record)))
    return b, calls


def test_openai_complete_sends_strict_schema_and_retries(monkeypatch):
    monkeypatch.setattr("ajaanai.llm.time.sleep", lambda s: None)
    replies = iter([httpx.Response(429), httpx.Response(200, json=completion(json.dumps({"ok": True})))])
    b, calls = openai_backend(lambda request: next(replies))
    r = b.complete(req())
    assert r.data == {"ok": True} and (r.input_tokens, r.output_tokens) == (7, 3) and len(calls) == 2
    sent = calls[1]
    assert str(sent.url) == "https://api.example/v1/chat/completions" and sent.headers["Authorization"] == "Bearer k"
    body = json.loads(sent.content)
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert body["max_completion_tokens"] == 100 and "reasoning_effort" not in body
    assert body["response_format"]["json_schema"] == {"name": "output", "strict": True, "schema": SCHEMA}


@pytest.mark.parametrize("reply", [completion('{"ok": true}', finish="length"), completion(None, refusal="no"),
                                   completion("not json")])
def test_openai_cut_off_refused_or_unparseable_gives_no_data(reply):
    b, _ = openai_backend(lambda request: httpx.Response(200, json=reply))
    r = b.complete(req())
    assert r.data is None and r.output_tokens == 3


def test_openai_reasoning_models_get_effort_and_room_to_reason():
    reasoning, plain = OpenAIBackend.body(req("gpt-5-mini", effort="medium")), OpenAIBackend.body(req("gpt-4.1", effort="medium"))
    assert reasoning["reasoning_effort"] == "medium" and reasoning["max_completion_tokens"] > 100
    assert "reasoning_effort" not in plain and plain["max_completion_tokens"] == 100


def test_openai_batch_cycle():
    state = {"status": "in_progress"}
    uploads = []

    def handler(request):
        path = request.url.path.removeprefix("/v1")
        if path == "/files":
            uploads.append(request.content)
            return httpx.Response(200, json={"id": "file-in"})
        if path == "/batches":
            assert json.loads(request.content) == {"input_file_id": "file-in", "endpoint": "/v1/chat/completions",
                                                   "completion_window": "24h"}
            return httpx.Response(200, json={"id": "batch_1"})
        if path == "/batches/batch_1":
            return httpx.Response(200, json={"status": state["status"], "output_file_id": "file-out",
                                             "error_file_id": "file-err",
                                             "request_counts": {"total": 3, "completed": 2, "failed": 1}})
        if path == "/files/file-out/content":
            lines = [{"custom_id": "p0", "response": {"status_code": 200, "body": completion('{"ok": true}')}},
                     {"custom_id": "p1", "response": {"status_code": 200, "body": completion("{}", finish="length")}}]
            return httpx.Response(200, text="\n".join(map(json.dumps, lines)) + "\n")
        if path == "/files/file-err/content":
            return httpx.Response(200, text=json.dumps({"custom_id": "p2", "response": {"status_code": 400, "body": {}},
                                                        "error": None}))
        raise AssertionError(path)

    b, _ = openai_backend(handler)
    assert b.submit([("p0", req()), ("p1", req()), ("p2", req())]) == "batch_1"
    assert b"name=\"purpose\"" in uploads[0] and b"batch" in uploads[0]
    first = json.loads(uploads[0].split(b"\r\n\r\n", 2)[2].split(b"\n")[0])
    assert first["custom_id"] == "p0" and first["url"] == "/v1/chat/completions" and first["body"]["model"] == "m"
    st = b.status("batch_1")
    assert not st.ended and (st.done, st.total) == (3, 3)
    state["status"] = "completed"
    assert b.status("batch_1").ended
    results = dict(b.results("batch_1"))
    assert results["p0"].data == {"ok": True} and results["p1"].data is None and results["p2"] is None


def test_gpt_6_models_are_reasoning_models():
    body = OpenAIBackend.body(req("gpt-6.1-sol", effort="high"))
    assert body["reasoning_effort"] == "high" and body["max_completion_tokens"] > 100

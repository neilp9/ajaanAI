"""Structured LLM calls for the pipeline's own helpers (synthesis, the responsiveness check, the eval
caller and judge), with Claude and OpenAI-compatible APIs as equal backends.

Every model is named with its provider: "anthropic:<model>" or "openai:<model>" (any OpenAI-compatible
chat API at OPENAI_BASE_URL). Callers build a provider-neutral `Request` (system prompt, one user
message, a JSON schema for the reply) and get back a `Reply` with the parsed JSON, or `data=None`
when the model was cut off, refused or returned something unparseable. Both backends also run
batches (Anthropic Message Batches / OpenAI Batch API: half price, finished within 24 h).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Iterator, Protocol

import httpx

log = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "openai")


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_spec(spec: str) -> ModelSpec:
    provider, sep, model = spec.strip().partition(":")
    if not sep or provider not in PROVIDERS or not model.strip():
        raise ValueError(f"Model {spec.strip()!r}: name the provider, as anthropic:<model> or openai:<model>")
    return ModelSpec(provider, model.strip())


def parse_specs(specs: str | list[str]) -> list[ModelSpec]:
    """Comma-separated, or a list of (comma-separated) specs."""
    items = specs.split(",") if isinstance(specs, str) else [x for s in specs for x in s.split(",")]
    return [parse_spec(x) for x in items if x.strip()]


@dataclass
class Request:
    model: str
    system: str
    user: str
    schema: dict
    max_tokens: int  # for the reply itself; backends add room for any hidden reasoning
    effort: str | None = None  # reasoning effort, where the model takes one
    fallback: bool = False  # let the provider retry a refusal on a fallback model, where it offers that


@dataclass
class Reply:
    data: dict | None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class BatchState:
    ended: bool
    done: int
    total: int
    succeeded: int
    errored: int


class Backend(Protocol):
    max_per_batch: int

    def complete(self, req: Request) -> Reply: ...

    def submit(self, items: list[tuple[str, Request]]) -> str: ...

    def status(self, batch_id: str) -> BatchState: ...

    def results(self, batch_id: str) -> Iterator[tuple[str, Reply | None]]:
        """(custom id, reply) per request; reply is None for a request that failed outright."""
        ...


def _loads(text: str | None) -> dict | None:
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


# --- Anthropic ------------------------------------------------------------------------------------

# Claude models that take an effort setting and the server-side refusal fallback.
_ANTHROPIC_EXTRAS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


class AnthropicBackend:
    max_per_batch = 50_000

    def __init__(self, api_key: str | None = None, client=None):
        if client is None:
            import anthropic

            # Falls back to the SDK's default credentials (env / `ant auth login` profile) when unset.
            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.client = client

    def _params(self, req: Request) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": req.schema}}
        if req.effort and req.model in _ANTHROPIC_EXTRAS:
            output_config["effort"] = req.effort
        return dict(model=req.model, max_tokens=req.max_tokens, system=req.system,
                    messages=[{"role": "user", "content": req.user}], output_config=output_config)

    @staticmethod
    def _reply(message) -> Reply:
        u = getattr(message, "usage", None)
        tokens = dict(input_tokens=getattr(u, "input_tokens", 0) or 0, output_tokens=getattr(u, "output_tokens", 0) or 0)
        if message.stop_reason != "end_turn":
            return Reply(None, **tokens)
        return Reply(_loads(next((b.text for b in message.content if b.type == "text"), "")), **tokens)

    def complete(self, req: Request) -> Reply:
        params = self._params(req)
        if req.fallback and req.model in _ANTHROPIC_EXTRAS:
            message = self.client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"],
                                                       fallbacks="default")
        else:
            message = self.client.messages.create(**params)
        return self._reply(message)

    def submit(self, items: list[tuple[str, Request]]) -> str:
        return self.client.messages.batches.create(
            requests=[{"custom_id": cid, "params": self._params(req)} for cid, req in items]).id

    def status(self, batch_id: str) -> BatchState:
        b = self.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        done = c.succeeded + c.errored + c.canceled + c.expired
        return BatchState(b.processing_status == "ended", done, done + c.processing, c.succeeded, c.errored)

    def results(self, batch_id: str) -> Iterator[tuple[str, Reply | None]]:
        for r in self.client.messages.batches.results(batch_id):
            yield r.custom_id, self._reply(r.result.message) if r.result.type == "succeeded" else None


# --- OpenAI-compatible -----------------------------------------------------------------------------

# OpenAI reasoning models, which take `reasoning_effort` (others reject it). Their reasoning counts
# toward max_completion_tokens, so they get this much room on top of the reply's own budget
# (gpt-5-mini spent ~450 reasoning tokens labelling one short conversation).
_OPENAI_REASONING = ("gpt-5", "gpt-6", "o1", "o3", "o4")
_REASONING_ROOM = 16_000
_BATCH_DONE = ("completed", "failed", "expired", "cancelled")


class OpenAIBackend:
    # The Batch API takes at most 50,000 requests and a 200 MB file; shaped prompts are a few KB each.
    max_per_batch = 20_000

    def __init__(self, base_url: str, api_key: str | None, http: httpx.Client | None = None, attempts: int = 4):
        self.base = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=600)
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.attempts = attempts

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        """One API call, retried with backoff on rate limits, server errors and dropped connections."""
        for attempt in range(self.attempts):
            try:
                r = self.http.request(method, self.base + path, headers=self.headers, **kw)
                if r.status_code != 429 and r.status_code < 500:
                    break
            except httpx.TransportError:
                if attempt == self.attempts - 1:
                    raise
            if attempt < self.attempts - 1:
                time.sleep(2 ** attempt * 5)
        r.raise_for_status()
        return r

    @staticmethod
    def body(req: Request) -> dict:
        reasoning = req.model.startswith(_OPENAI_REASONING)
        body = {
            "model": req.model,
            "max_completion_tokens": req.max_tokens + (_REASONING_ROOM if reasoning else 0),
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "output", "strict": True, "schema": req.schema}},
        }
        if req.effort and reasoning:
            body["reasoning_effort"] = req.effort
        return body

    @staticmethod
    def _reply(completion: dict) -> Reply:
        u = completion.get("usage") or {}
        tokens = dict(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0))
        choice = completion["choices"][0]
        msg = choice["message"]
        if choice.get("finish_reason") != "stop" or msg.get("refusal"):
            return Reply(None, **tokens)
        return Reply(_loads(msg.get("content")), **tokens)

    def complete(self, req: Request) -> Reply:
        return self._reply(self._call("POST", "/chat/completions", json=self.body(req)).json())

    def submit(self, items: list[tuple[str, Request]]) -> str:
        lines = "\n".join(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                                      "body": self.body(req)}) for cid, req in items)
        f = self._call("POST", "/files", data={"purpose": "batch"},
                       files={"file": ("batch.jsonl", lines.encode(), "application/jsonl")}).json()
        return self._call("POST", "/batches", json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions",
                                                     "completion_window": "24h"}).json()["id"]

    def status(self, batch_id: str) -> BatchState:
        b = self._call("GET", f"/batches/{batch_id}").json()
        if b["status"] == "failed":
            log.warning("batch %s failed: %s", batch_id, b.get("errors"))
        c = b.get("request_counts") or {}
        ok, bad, total = c.get("completed", 0), c.get("failed", 0), c.get("total", 0)
        return BatchState(b["status"] in _BATCH_DONE, ok + bad, total, ok, bad)

    def results(self, batch_id: str) -> Iterator[tuple[str, Reply | None]]:
        b = self._call("GET", f"/batches/{batch_id}").json()
        for file_id in (b.get("output_file_id"), b.get("error_file_id")):
            if not file_id:
                continue
            for line in self._call("GET", f"/files/{file_id}/content").text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                resp = row.get("response") or {}
                ok = resp.get("status_code") == 200 and not row.get("error")
                yield row["custom_id"], self._reply(resp["body"]) if ok else None


def make_backend(spec: ModelSpec, settings) -> Backend:
    """The backend for a model spec, with keys from the settings. Fails fast on a missing key."""
    if spec.provider == "anthropic":
        return AnthropicBackend(settings.anthropic_api_key)
    if not settings.openai_api_key and "api.openai.com" in settings.openai_base_url:
        raise SystemExit(f"{spec.name} needs OPENAI_API_KEY (or OPENAI_BASE_URL for another OpenAI-compatible API)")
    return OpenAIBackend(settings.openai_base_url, settings.openai_api_key)

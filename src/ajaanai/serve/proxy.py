"""OpenAI-compatible proxy that sits in front of *any* chat model.

ElevenLabs (or the `ajaanai chat` CLI) calls POST /v1/chat/completions here. For each turn it:
  1. decides whether the model should think (REASONING_MODE off | on | auto);
  2. retrieves reference passages (RAG) and builds the system prompt;
  3. streams the upstream model, stripping reasoning (reasoning_content or <think> tags);
  4. while the caller would otherwise hear silence, streams short filler phrases.

The upstream is any OpenAI-style /chat/completions endpoint: the vLLM server in the same
Modal container, an OpenAI fine-tune, Together, a local llama.cpp, etc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..dataset.persona import FILLERS_FIRST, FILLERS_REPEAT, RAG_PREAMBLE, SYSTEM_PROMPT
from ..models import ModelProfile

log = logging.getLogger(__name__)

# --- reasoning decision ------------------------------------------------------------

_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|thanks?|thank you|ok(ay)?|yes|no|yeah|sure|bye|goodbye|good (morning|evening|night)"
    r"|how are you|can you hear me|are you there)\b[\s\S]{0,40}$",
    re.IGNORECASE,
)
_DEEP = re.compile(
    r"\b(why|how (do|can|should)|what (is|does|should)|explain|difference|meaning|mean by|practice|"
    r"suffer|kamma|karma|rebirth|not-?self|anatt|jh[aā]na|concentrat|meditat|death|dying|grief|"
    r"anger|fear|desire|craving|nibb[aā]na|emptiness|precept|forgive|struggl)",
    re.IGNORECASE,
)


def should_think(mode: str, user_text: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    text = user_text.strip()
    if not text or _SMALL_TALK.match(text):
        return False
    words = len(text.split())
    return words >= 25 or (words >= 6 and bool(_DEEP.search(text)))


# --- think-tag stripping -------------------------------------------------------------


class ThinkStripper:
    """Removes <think>…</think> spans from a streamed text, tolerant of tags split across chunks.

    With `expect_reasoning`, leading output is held back until we learn whether it is reasoning:
    a close tag (template opened <think> in the prompt), an open tag, or enough plain text to
    conclude the model isn't reasoning inline.
    """

    def __init__(self, open_tag: str = "<think>", close_tag: str = "</think>",
                 expect_reasoning: bool = False, holdback_chars: int = 400):
        self.open, self.close = open_tag, close_tag
        self.inside = False
        self.buf = ""
        self.undecided = expect_reasoning
        self.holdback = holdback_chars

    def resolve(self) -> None:
        """Reasoning arrives out-of-band (reasoning_content), so content is all visible."""
        self.undecided = False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = []
        while True:
            if self.undecided:
                ci, oi = self.buf.find(self.close), self.buf.find(self.open)
                if ci != -1 and (oi == -1 or ci < oi):  # implicit open: drop everything up to </think>
                    self.buf = self.buf[ci + len(self.close):]
                    self.undecided = False
                    continue
                if oi != -1:
                    out.append(self.buf[:oi])
                    self.buf = self.buf[oi + len(self.open):]
                    self.inside, self.undecided = True, False
                    continue
                if len(self.buf) > self.holdback:
                    self.undecided = False
                    continue
                break
            if self.inside:
                i = self.buf.find(self.close)
                if i == -1:
                    self.buf = self.buf[-(len(self.close) - 1):]
                    break
                self.buf = self.buf[i + len(self.close):]
                self.inside = False
                continue
            i = self.buf.find(self.open)
            if i == -1:
                keep = _partial_suffix(self.buf, self.open)
                out.append(self.buf[: len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                break
            out.append(self.buf[:i])
            self.buf = self.buf[i + len(self.open):]
            self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


def _partial_suffix(s: str, tag: str) -> int:
    for n in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:n]):
            return n
    return 0


# --- request building ---------------------------------------------------------------------


@dataclass
class ProxyConfig:
    upstream_base_url: str  # e.g. http://127.0.0.1:8000/v1
    upstream_model: str
    upstream_api_key: str | None
    profile: ModelProfile
    reasoning_mode: str = "auto"
    reasoning_budget_tokens: int = 1024
    max_answer_tokens: int = 400
    filler_mode: str = "when_thinking"
    filler_interval_s: float = 4.0
    fillers_first: tuple[str, ...] = tuple(FILLERS_FIRST)
    fillers_repeat: tuple[str, ...] = tuple(FILLERS_REPEAT)
    send_template_kwargs: bool = True  # only our own vLLM understands chat_template_kwargs
    bearer_secret: str | None = None


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


def build_upstream_body(cfg: ProxyConfig, body: dict, think: bool, references: str | None) -> dict:
    incoming = body.get("messages", [])
    extra_system = [m["content"] for m in incoming if m.get("role") == "system" and isinstance(m.get("content"), str)]
    system = SYSTEM_PROMPT
    if references:
        system += "\n\n" + RAG_PREAMBLE + references
    if extra_system:  # e.g. ElevenLabs agent prompt / tool instructions
        system += "\n\nAdditional instructions from the voice platform:\n" + "\n".join(extra_system)
    messages = [{"role": "system", "content": system}] + [m for m in incoming if m.get("role") != "system"]
    out = {
        "model": cfg.upstream_model,
        "messages": messages,
        "stream": True,
        "max_tokens": cfg.max_answer_tokens + (cfg.reasoning_budget_tokens if think else 0),
        "temperature": body.get("temperature", 0.7),
    }
    if cfg.send_template_kwargs and cfg.profile.chat_template_kwargs(think):
        out["chat_template_kwargs"] = cfg.profile.chat_template_kwargs(think)
    for k in ("tools", "tool_choice"):
        if k in body:
            out[k] = body[k]
    return out


# --- streaming -----------------------------------------------------------------------------


@dataclass
class Delta:
    content: str = ""
    reasoning: str = ""
    tool_calls: list | None = None
    finish_reason: str | None = None


async def upstream_stream(client: httpx.AsyncClient, cfg: ProxyConfig, body: dict) -> AsyncIterator[Delta]:
    headers = {"Authorization": f"Bearer {cfg.upstream_api_key}"} if cfg.upstream_api_key else {}
    url = cfg.upstream_base_url.rstrip("/") + "/chat/completions"
    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(f"upstream {resp.status_code}: {(await resp.aread())[:300]!r}")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                d = choice.get("delta", {})
                yield Delta(
                    content=d.get("content") or "",
                    reasoning=d.get("reasoning_content") or d.get("reasoning") or "",
                    tool_calls=d.get("tool_calls"),
                    finish_reason=choice.get("finish_reason"),
                )


class FillerPicker:
    def __init__(self, first: tuple[str, ...], repeat: tuple[str, ...], rng: random.Random | None = None):
        self.first, self.repeat, self.rng, self.last = first, repeat, rng or random.Random(), None

    def _pick(self, pool):
        choices = [p for p in pool if p != self.last] or list(pool)
        self.last = self.rng.choice(choices)
        return self.last

    def first_filler(self) -> str:
        return self._pick(self.first)

    def repeat_filler(self) -> str:
        return self._pick(self.repeat)


async def spoken_stream(
    deltas: AsyncIterator[Delta], cfg: ProxyConfig, think: bool,
    clock: Callable[[], float] = time.monotonic, rng: random.Random | None = None,
) -> AsyncIterator[Delta]:
    """Visible deltas for the caller: reasoning removed, fillers inserted during silences."""
    want_fillers = cfg.filler_mode == "always" or (cfg.filler_mode == "when_thinking" and think)
    picker = FillerPicker(cfg.fillers_first, cfg.fillers_repeat, rng)
    stripper = ThinkStripper(cfg.profile.think_open, cfg.profile.think_close,
                             expect_reasoning=think and cfg.profile.thinking != "none")
    answering = False
    last_spoken = clock()
    if want_fillers:
        yield Delta(content=picker.first_filler())
        last_spoken = clock()

    it = deltas.__aiter__()
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(it.__anext__())
            timeout = None
            if want_fillers and not answering:
                timeout = max(0.0, cfg.filler_interval_s - (clock() - last_spoken))
            done, _ = await asyncio.wait({pending}, timeout=timeout)
            if not done:
                yield Delta(content=picker.repeat_filler())
                last_spoken = clock()
                continue
            try:
                d = pending.result()
            except StopAsyncIteration:
                break
            pending = None
            if d.reasoning:
                stripper.resolve()
            visible = stripper.feed(d.content) if d.content else ""
            if visible:
                answering = True
                yield Delta(content=visible)
                last_spoken = clock()
            if d.tool_calls:
                yield Delta(tool_calls=d.tool_calls)
            if d.finish_reason:
                tail = stripper.flush()
                if tail:
                    yield Delta(content=tail)
                yield Delta(finish_reason=d.finish_reason)
                return
        tail = stripper.flush()
        if tail:
            yield Delta(content=tail)
        yield Delta(finish_reason="stop")
    finally:
        if pending is not None and not pending.done():
            pending.cancel()


def sse_chunk(cid: str, model: str, delta: Delta, first: bool = False) -> str:
    d: dict = {}
    if first:
        d["role"] = "assistant"
    if delta.content:
        d["content"] = delta.content
    if delta.tool_calls:
        d["tool_calls"] = delta.tool_calls
    payload = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
               "choices": [{"index": 0, "delta": d, "finish_reason": delta.finish_reason}]}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# --- FastAPI app -----------------------------------------------------------------------------


def create_app(cfg: ProxyConfig, retrieve: Callable[[str], str | None] | None = None,
               on_warm: Callable[[], None] | None = None):
    app = FastAPI(title="ajaanAI proxy")
    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    def check_auth(request: Request):
        if cfg.bearer_secret and request.headers.get("authorization") != f"Bearer {cfg.bearer_secret}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.api_route("/warm", methods=["GET", "POST"])
    async def warm():
        if on_warm:
            await asyncio.to_thread(on_warm)
        return {"ok": True}

    @app.post("/elevenlabs/init")
    async def elevenlabs_init(request: Request):
        """Conversation-initiation webhook: called when a call connects — we use it to wake the GPU."""
        if on_warm:
            asyncio.get_running_loop().run_in_executor(None, on_warm)
        return {"type": "conversation_initiation_client_data", "dynamic_variables": {}}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": cfg.upstream_model, "object": "model"}]}

    # ElevenLabs may treat the configured URL as a server root or as an OpenAI base_url;
    # answer on every path either interpretation produces.
    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    @app.post("/v1/v1/chat/completions")
    async def chat(request: Request):
        check_auth(request)
        body = await request.json()
        user_text = last_user_text(body.get("messages", []))
        think = cfg.profile.thinking != "none" and should_think(cfg.reasoning_mode, user_text)
        if cfg.profile.thinking == "always":
            think = True
        references = await asyncio.to_thread(retrieve, user_text) if (retrieve and user_text) else None
        up_body = build_upstream_body(cfg, body, think, references)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        stream = spoken_stream(upstream_stream(client, cfg, up_body), cfg, think)

        if body.get("stream", False):
            async def sse():
                first = True
                try:
                    async for d in stream:
                        yield sse_chunk(cid, cfg.upstream_model, d, first)
                        first = False
                except Exception as e:  # keep the call alive with a graceful line
                    log.exception("upstream failed")
                    yield sse_chunk(cid, cfg.upstream_model,
                                    Delta(content="I'm sorry, I lost my train of thought there. Could you say that again?"), first)
                    yield sse_chunk(cid, cfg.upstream_model, Delta(finish_reason="stop"))
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming: no fillers, just the answer.
        cfg_quiet = ProxyConfig(**{**cfg.__dict__, "filler_mode": "off"})
        text, finish = [], "stop"
        async for d in spoken_stream(upstream_stream(client, cfg_quiet, up_body), cfg_quiet, think):
            text.append(d.content)
            finish = d.finish_reason or finish
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": int(time.time()), "model": cfg.upstream_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                         "finish_reason": finish}],
        })

    return app

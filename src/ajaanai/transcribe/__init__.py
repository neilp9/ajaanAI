"""Pluggable transcription. Choose with TRANSCRIBE_PROVIDER (+ that provider's API key)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Callable

from ..catalog import Catalog, Doc
from ..config import Settings
from .base import GLOSSARY, Transcriber, absolute_audio_url, paragraphize
from .providers import AssemblyAI, Deepgram, ElevenLabsScribe, ModalWhisper, OpenAITranscriber

log = logging.getLogger(__name__)

_KEYS = {
    "elevenlabs": "elevenlabs_api_key",
    "openai": "openai_api_key",
    "deepgram": "deepgram_api_key",
    "assemblyai": "assemblyai_api_key",
    "modal_whisper": None,
}


def get_transcriber(settings: Settings, provider: str | None = None, model: str | None = None) -> Transcriber:
    provider = provider or settings.transcribe_provider
    if provider not in _KEYS:
        raise SystemExit(f"Unknown transcription provider {provider!r}. Choose from: {', '.join(_KEYS)}")
    key_name = _KEYS[provider]
    if key_name:
        settings.require(key_name)
    model = model or settings.transcribe_model
    key = getattr(settings, key_name) if key_name else None
    return {
        "elevenlabs": lambda: ElevenLabsScribe(key, model),
        "openai": lambda: OpenAITranscriber(key, model),
        "deepgram": lambda: Deepgram(key, model),
        "assemblyai": lambda: AssemblyAI(key, model),
        "modal_whisper": lambda: ModalWhisper(model),
    }[provider]()


def transcribe_docs(cat: Catalog, docs: list[Doc], settings: Settings, provider: str | None = None,
                    model: str | None = None,
                    checkpoint: Callable[[], None] | None = None) -> tuple[list[str], list[str]]:
    """Transcribe docs concurrently; returns (succeeded ids, failed ids)."""
    if not docs:
        return [], []
    tr = get_transcriber(settings, provider, model)
    ok, failed = [], []

    def work(doc: Doc) -> str:
        url = absolute_audio_url(settings.site_base_url, doc.mp3_url)
        return paragraphize(tr.transcribe(url, GLOSSARY))

    with ThreadPoolExecutor(max_workers=settings.transcribe_concurrency) as pool:
        futures = {pool.submit(work, d): d for d in docs}
        for fut in as_completed(futures):
            doc = futures[fut]
            try:
                text = fut.result()
                if not text.strip():
                    raise ValueError("empty transcript")
                cat.set_text(doc.id, text, f"asr:{tr.name}")
                ok.append(doc.id)
                log.info("transcribed %s (%s)", doc.id, tr.name)
            except Exception as e:
                cat.mark_failed(doc.id, str(e))
                failed.append(doc.id)
                log.warning("transcription failed for %s: %s", doc.id, e)
            if checkpoint and (len(ok) + len(failed)) % 10 == 0:
                checkpoint()
    return ok, failed


def transcribe_pending(cat: Catalog, settings: Settings, *, sources: list[str] | None = None,
                       grace_days: int | None = None, limit: int | None = None,
                       provider: str | None = None, model: str | None = None,
                       today: date | None = None, retry_failed: bool = False,
                       checkpoint: Callable[[], None] | None = None) -> tuple[list[str], list[str]]:
    grace = settings.transcribe_grace_days if grace_days is None else grace_days
    docs = cat.ready_for_asr(grace, sources=sources, limit=limit, today=today, include_failed=retry_failed)
    log.info("%d docs ready for transcription", len(docs))
    return transcribe_docs(cat, docs, settings, provider, model, checkpoint)

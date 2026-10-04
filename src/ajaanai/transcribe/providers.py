"""ASR providers. Each takes a public audio URL and returns plain transcript text.

Providers that can fetch a URL themselves (ElevenLabs, Deepgram, AssemblyAI) never touch
the audio locally; OpenAI needs an upload, so we download (and split if > 24 MB).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from .base import TranscriptionError

TIMEOUT = httpx.Timeout(900.0, connect=20.0)


def _check(resp: httpx.Response) -> dict:
    if resp.status_code >= 400:
        raise TranscriptionError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


class ElevenLabsScribe:
    name = "elevenlabs"
    default_model = "scribe_v2"

    def __init__(self, api_key: str, model: str | None = None):
        self.key, self.model = api_key, model or self.default_model

    def transcribe(self, audio_url: str, glossary: list[str]) -> str:
        data = [("model_id", self.model), ("cloud_storage_url", audio_url), ("language_code", "en"),
                ("tag_audio_events", "false"), ("diarize", "false")]
        data += [("keyterms", t) for t in glossary[:1000] if len(t) <= 50]
        resp = httpx.post("https://api.elevenlabs.io/v1/speech-to-text", data=data,
                          headers={"xi-api-key": self.key}, timeout=TIMEOUT)
        return _check(resp)["text"]


class Deepgram:
    name = "deepgram"
    default_model = "nova-3"

    def __init__(self, api_key: str, model: str | None = None):
        self.key, self.model = api_key, model or self.default_model

    def transcribe(self, audio_url: str, glossary: list[str]) -> str:
        params = [("model", self.model), ("smart_format", "true"), ("paragraphs", "true"),
                  ("language", "en")] + [("keyterm", t) for t in glossary[:100]]
        resp = httpx.post("https://api.deepgram.com/v1/listen", params=params, json={"url": audio_url},
                          headers={"Authorization": f"Token {self.key}"}, timeout=TIMEOUT)
        alt = _check(resp)["results"]["channels"][0]["alternatives"][0]
        paras = alt.get("paragraphs", {}).get("paragraphs")
        if paras:
            return "\n\n".join(" ".join(s["text"] for s in p["sentences"]) for p in paras)
        return alt["transcript"]


class AssemblyAI:
    name = "assemblyai"
    default_model = "universal"
    base = "https://api.assemblyai.com/v2"

    def __init__(self, api_key: str, model: str | None = None, poll_s: float = 5.0):
        self.key, self.model, self.poll_s = api_key, model or self.default_model, poll_s

    def transcribe(self, audio_url: str, glossary: list[str]) -> str:
        h = {"authorization": self.key}
        job = _check(httpx.post(f"{self.base}/transcript", headers=h, timeout=60, json={
            "audio_url": audio_url, "speech_model": self.model, "language_code": "en",
            "word_boost": glossary[:1000], "boost_param": "default",
        }))
        while True:
            t = _check(httpx.get(f"{self.base}/transcript/{job['id']}", headers=h, timeout=60))
            if t["status"] == "completed":
                break
            if t["status"] == "error":
                raise TranscriptionError(t.get("error", "assemblyai error"))
            time.sleep(self.poll_s)
        paras = _check(httpx.get(f"{self.base}/transcript/{job['id']}/paragraphs", headers=h, timeout=60))
        return "\n\n".join(p["text"] for p in paras.get("paragraphs", [])) or t["text"]


class OpenAITranscriber:
    name = "openai"
    default_model = "gpt-4o-transcribe"
    max_bytes = 24 * 1024 * 1024

    def __init__(self, api_key: str, model: str | None = None):
        self.key, self.model = api_key, model or self.default_model

    def _one(self, path: Path, prompt: str) -> str:
        with path.open("rb") as f:
            resp = httpx.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.key}"},
                data={"model": self.model, "prompt": prompt, "response_format": "json", "language": "en"},
                files={"file": (path.name, f, "audio/mpeg")}, timeout=TIMEOUT,
            )
        return _check(resp)["text"]

    def transcribe(self, audio_url: str, glossary: list[str]) -> str:
        prompt = "Dhamma talk by Thanissaro Bhikkhu. Terms: " + ", ".join(glossary[:60])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "audio.mp3"
            with httpx.stream("GET", audio_url, timeout=TIMEOUT, follow_redirects=True) as r:
                r.raise_for_status()
                with src.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            if src.stat().st_size <= self.max_bytes:
                return self._one(src, prompt)
            return "\n\n".join(self._one(p, prompt) for p in split_audio(src, Path(tmp)))


class ModalWhisper:
    """Self-hosted faster-whisper on a Modal GPU (see ajaanai.modal_app.whisper_transcribe)."""

    name = "modal_whisper"
    default_model = "large-v3-turbo"

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    def transcribe(self, audio_url: str, glossary: list[str]) -> str:
        import modal

        fn = modal.Function.from_name("ajaanai", "whisper_transcribe")
        return fn.remote(audio_url, self.model, glossary)


def split_audio(src: Path, out_dir: Path, segment_s: int = 1200) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise TranscriptionError("audio > 24MB needs ffmpeg to split (brew install ffmpeg)")
    pattern = out_dir / "part%03d.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(src), "-f", "segment", "-segment_time",
                    str(segment_s), "-c", "copy", str(pattern)], check=True)
    return sorted(out_dir.glob("part*.mp3"))

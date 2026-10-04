from __future__ import annotations

import re
from typing import Protocol
from urllib.parse import quote

# Pali / Thai Forest vocabulary that generic ASR tends to mangle. Passed as a prompt or
# key-terms list to providers that support biasing.
GLOSSARY = [
    "Dhamma", "Vinaya", "jhāna", "jhana", "samādhi", "samadhi", "vipassanā", "satipaṭṭhāna",
    "sati", "sampajañña", "saṅkhāra", "sankhara", "dukkha", "anicca", "anattā", "anatta",
    "nibbāna", "nibbana", "kamma", "karma", "mettā", "metta", "karuṇā", "muditā", "upekkhā",
    "brahmavihāra", "pīti", "sukha", "vitakka", "vicāra", "khandha", "taṇhā", "tanha", "avijjā",
    "paṭicca samuppāda", "saṁvega", "samvega", "pasāda", "dāna", "sīla", "paññā", "bhāvanā",
    "Tathāgata", "arahant", "stream-entry", "Sangha", "sutta", "Pāli", "Ajaan Lee", "Ajaan Fuang",
    "Ajaan Mun", "Ajaan Suwat", "Ajaan Chah", "Wat Dhammasathit", "Metta Forest Monastery",
    "Thanissaro", "Ṭhānissaro", "Buddho", "Rayong",
]

AUDIO_SAFE = "/()&',!:;+$@=~"


class Transcriber(Protocol):
    name: str
    default_model: str

    def transcribe(self, audio_url: str, glossary: list[str]) -> str: ...


class TranscriptionError(RuntimeError):
    pass


def absolute_audio_url(base_url: str, mp3_path: str) -> str:
    """Site paths contain spaces etc. ('/Archive/y2005/051231 Proving the Teachings.mp3')."""
    if mp3_path.startswith("http"):
        return mp3_path
    return base_url.rstrip("/") + quote(mp3_path, safe=AUDIO_SAFE)


_SENT_END = re.compile(r"(?<=[.!?])\s+")


def paragraphize(text: str, target_words: int = 120) -> str:
    """ASR returns walls of text; break into readable ~target-length paragraphs at sentence ends."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if "\n\n" in text:  # provider already gave paragraphs
        return text
    paras, cur, n = [], [], 0
    for sent in _SENT_END.split(text):
        cur.append(sent)
        n += len(sent.split())
        if n >= target_words:
            paras.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)

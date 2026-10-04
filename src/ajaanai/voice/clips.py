"""Prepare a Professional Voice Clone training set from evening talk recordings.

This only *prepares* audio. The clone itself must be created and verified on Ajaan Geoff's
(or the monastery's) own ElevenLabs account and shared with ours — see CONSENT.md.

Selection: recent talks (newer recordings are cleaner), lead-in trimmed, long silences removed,
loudness normalised to ElevenLabs' recommended range, exported as ~30-minute MP3s.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from ..catalog import Catalog
from ..config import Settings
from ..transcribe.base import absolute_audio_url

FILTERS = ",".join([
    "highpass=f=70",
    "silenceremove=stop_periods=-1:stop_duration=0.8:stop_threshold=-40dB",
    "loudnorm=I=-20:TP=-3:LRA=11",
])


def _duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip() or 0)


def prepare_pvc_clips(cat: Catalog, settings: Settings, out_dir: Path, hours: float = 2.0,
                      since_year: int = 2023, segment_minutes: int = 30) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required (brew install ffmpeg)")
    out_dir.mkdir(parents=True, exist_ok=True)
    talks = [d for d in cat.iter_with_text(sources=["evening"])
             if d.mp3_url and d.date and int(d.date[:4]) >= since_year]
    talks.sort(key=lambda d: d.date, reverse=True)
    total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        cleaned = []
        for doc in talks:
            if total >= hours * 3600:
                break
            src = Path(tmp) / "src.mp3"
            with httpx.stream("GET", absolute_audio_url(settings.site_base_url, doc.mp3_url), timeout=300,
                              follow_redirects=True) as r:
                r.raise_for_status()
                with src.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            dst = Path(tmp) / f"clean_{len(cleaned):03d}.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "15", "-i", str(src), "-af", FILTERS,
                            "-ac", "1", "-ar", "44100", str(dst)], check=True)
            total += _duration(dst)
            cleaned.append(dst)
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in cleaned))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-f", "segment", "-segment_time", str(segment_minutes * 60), "-c:a", "libmp3lame",
                        "-b:a", "192k", str(out_dir / "pvc_%02d.mp3")], check=True)
    produced = sorted(out_dir.glob("pvc_*.mp3"))
    (out_dir / "README.txt").write_text(
        f"{len(cleaned)} evening talks since {since_year}, ~{total / 3600:.1f} h after silence removal.\n"
        "For a Professional Voice Clone on the speaker's own ElevenLabs account, with his permission.\n"
        "Source: dhammatalks.org (CC BY-NC 4.0).\n")
    return produced

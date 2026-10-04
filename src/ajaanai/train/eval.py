"""Compare model "arms" on held-out questions, judged by Claude.

An arm is any OpenAI-compatible endpoint — usually the deployed proxy (fine-tune + RAG), the proxy
with RAG off, or the raw base model — so the same eval works whatever LLM is behind it.
Each answer is scored 1-5 for *style* (sounds like him) and *fidelity* (consistent with the
reference passage, no invented doctrine).
"""

from __future__ import annotations

import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anthropic
import httpx

JUDGE_SYSTEM = """You grade answers from a voice assistant trained on the talks of Thanissaro \
Bhikkhu. You get the caller's question, a reference passage of his actual teaching on it, and the \
assistant's answer. Score:
- style (1-5): does it sound like him — plain, direct, practical, his characteristic analogies — \
and suit a spoken phone call (concise, no lists)?
- fidelity (1-5): is it consistent with the reference and with his teachings generally? Penalise \
invented doctrine, misattributed claims, or contradictions. Don't require it to repeat the reference.
Be strict: 5 means you would not be able to tell it from him."""

SCHEMA = {
    "type": "object",
    "properties": {
        "style": {"type": "integer"},
        "fidelity": {"type": "integer"},
        "note": {"type": "string"},
    },
    "required": ["style", "fidelity", "note"],
    "additionalProperties": False,
}


@dataclass
class Arm:
    name: str
    base_url: str  # .../v1
    model: str = "ajaan"
    api_key: str | None = None

    @classmethod
    def parse(cls, spec: str, default_key: str | None) -> "Arm":
        """'name=https://host/v1' or 'name=https://host/v1|model'"""
        name, _, rest = spec.partition("=")
        url, _, model = rest.partition("|")
        return cls(name, url, model or "ajaan", default_key)


def ask(arm: Arm, question: str) -> str:
    h = {"Authorization": f"Bearer {arm.api_key}"} if arm.api_key else {}
    r = httpx.post(arm.base_url.rstrip("/") + "/chat/completions", headers=h, timeout=180, json={
        "model": arm.model, "stream": False, "messages": [{"role": "user", "content": question}],
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def judge(client: anthropic.Anthropic, model: str, q: dict, answer: str) -> dict | None:
    msg = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"Question:\n{q['question']}\n\nReference passage:\n{q['reference']}\n\nAnswer:\n{answer}"}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if msg.stop_reason != "end_turn":
        return None
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return json.loads(text)


def run_eval(questions_path: Path, arms: list[Arm], judge_model: str, anthropic_key: str,
             n: int = 100, out_dir: Path | None = None) -> dict:
    qs = [json.loads(line) for line in questions_path.open()]
    random.Random(0).shuffle(qs)
    qs = qs[:n]
    client = anthropic.Anthropic(api_key=anthropic_key)
    rows = []

    def one(args):
        arm, q = args
        try:
            ans = ask(arm, q["question"])
            score = judge(client, judge_model, q, ans)
        except Exception as e:
            return {"arm": arm.name, "question": q["question"], "error": str(e)}
        return {"arm": arm.name, "question": q["question"], "answer": ans, **(score or {"error": "judge declined"})}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, [(a, q) for a in arms for q in qs]))

    summary = {}
    for arm in arms:
        scored = [r for r in rows if r["arm"] == arm.name and "style" in r]
        summary[arm.name] = {
            "n": len(scored),
            "errors": sum(1 for r in rows if r["arm"] == arm.name and "error" in r),
            "style": round(statistics.mean(r["style"] for r in scored), 2) if scored else None,
            "fidelity": round(statistics.mean(r["fidelity"] for r in scored), 2) if scored else None,
        }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

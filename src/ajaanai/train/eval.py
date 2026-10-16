"""Compare model "arms" on whole held-out conversations, judged by an LLM.

An arm is any OpenAI-compatible endpoint — usually the deployed proxy (fine-tune + RAG), the proxy
with RAG off, or the raw base model — so the same eval works whatever LLM is behind it.

Each eval item is a held-out conversation opening (eval_questions.jsonl: the real first caller line,
the reference passage and the channel). The arm answers; then an LLM (EVAL_CALLER_MODEL) plays the
caller, responding to what the arm actually said (follow-ups, doubts, answers to its questions), for
up to `turns` exchanges. Replaying the dataset's scripted caller lines wouldn't work: they answer his original
sentences, not the arm's.

The judge scores the whole transcript 1-5 for *style* (sounds like him), *fidelity* (consistent with
the reference, no invented doctrine), *brevity* (one point at a time, no lecturing) and
*interactivity* (responds to what was just said, builds on earlier turns, hands the conversation
back). Reply length, questions back and markdown use are also measured directly.

Judges are configurable (JUDGE_MODEL, `--judge`): "anthropic:<model>" or "openai:<model>" (any
OpenAI-compatible chat API). Several judges can score the same transcripts; judges from different
model families check whether a judge favours replies that sound like its own family's. `rejudge`
re-scores saved transcripts and reports how the judges agree.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from ..llm import Backend, ModelSpec, Request

CHANNEL_NAMES = {"voice": "voice call", "text": "text chat"}

CALLER_SYSTEM = """You play a person talking with an AI trained on the talks of Thanissaro Bhikkhu, \
a Buddhist monk, over a {channel}. You opened with the first line of the transcript. Continue as \
that person: respond to what the AI just said — a follow-up, a doubt or pushback, something about \
your own situation, a request for an example, or an answer if it asked you something. Stay in \
character; don't evaluate the AI or steer it toward a particular teaching. Keep lines short \
(usually under 25 words). {style}
If the conversation has reached a natural close, set `end` to true and leave `line` empty."""

CALLER_STYLE = {
    "voice": "Write it as transcribed speech: plain spoken phrasing.",
    "text": "Write it as typed chat: short and casual, sometimes lowercase.",
}

CALLER_SCHEMA = {
    "type": "object",
    "properties": {"line": {"type": "string"}, "end": {"type": "boolean"}},
    "required": ["line", "end"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You grade a whole conversation between a person and an AI trained on the talks of \
Thanissaro Bhikkhu (Ajaan Geoff). You get the channel (voice call or text chat), a reference passage \
of his actual teaching on the opening topic, and the transcript. Score the AI across the whole \
conversation, 1-5 each:
- style: does it sound like him — plain, direct, warm but unsentimental, practical, his kind of \
everyday analogies? Plain speech suited to the channel (no lists, headings or markdown)?
- fidelity: is it consistent with the reference and with his teachings generally? Penalise invented \
doctrine, misattributed claims, or contradictions. Don't require it to repeat the reference.
- brevity: one point at a time — usually one to three sentences a reply, longer only when asked? \
Penalise lecturing, monologues, and piling several points into one reply.
- interactivity: does each reply respond to what the person just said, build on earlier turns, and \
hand the conversation back (sometimes with a short question turning them toward their own \
experience)? Penalise ignoring the person, repeating itself, or generic replies that would fit any \
message.
Be strict: 5 means you could not tell it from him in a real conversation."""

SCORES = ("style", "fidelity", "brevity", "interactivity")

SCHEMA = {
    "type": "object",
    "properties": {**{k: {"type": "integer"} for k in SCORES}, "note": {"type": "string"}},
    "required": [*SCORES, "note"],
    "additionalProperties": False,
}

_MARKDOWN = re.compile(r"(?m)^\s*(?:[-*•]\s|#{1,6}\s|\d+[.)]\s)|\*\*|__|(?<![*\w])\*[^*\s][^*\n]*\*(?![*\w])")


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

    def model_for(self, channel: str) -> str:
        """Our proxy picks the channel from the model name; an explicit model is sent as given."""
        return f"ajaan-{channel}" if self.model == "ajaan" else self.model


def ask(arm: Arm, messages: list[dict], channel: str) -> str:
    h = {"Authorization": f"Bearer {arm.api_key}"} if arm.api_key else {}
    # follow_redirects: Modal answers a slow request (e.g. a GPU cold start) with 303 -> result URL
    r = httpx.post(arm.base_url.rstrip("/") + "/chat/completions", headers=h, timeout=600, follow_redirects=True, json={
        "model": arm.model_for(channel), "stream": False, "messages": messages,
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def transcript(messages: list[dict]) -> str:
    return "\n\n".join(f"{'Person' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in messages)


def caller_line(backend: Backend, model: str, messages: list[dict], channel: str) -> str | None:
    """The simulated caller's next line, or None when they'd end the conversation."""
    out = backend.complete(Request(
        model=model,
        system=CALLER_SYSTEM.format(channel=CHANNEL_NAMES[channel], style=CALLER_STYLE[channel]),
        user=f"Transcript so far:\n\n{transcript(messages)}\n\nYour next line?",
        schema=CALLER_SCHEMA,
        max_tokens=1024,
    )).data
    if out is None or out["end"] or not out["line"].strip():
        return None
    return out["line"].strip()


def converse(arm: Arm, q: dict, turns: int, next_line: Callable[[list[dict], str], str | None],
             ask_fn: Callable[[Arm, list[dict], str], str] = ask) -> list[dict]:
    """Run one conversation: the real opening line, then the simulated caller, up to `turns` replies."""
    channel = q.get("channel", "voice")
    messages = [{"role": "user", "content": q["question"]}]
    for t in range(turns):
        messages.append({"role": "assistant", "content": ask_fn(arm, messages, channel)})
        if t == turns - 1 or (line := next_line(messages, channel)) is None:
            break
        messages.append({"role": "user", "content": line})
    return messages


def judge_prompt(q: dict, messages: list[dict]) -> str:
    return (f"Channel: {CHANNEL_NAMES[q.get('channel', 'voice')]}\n\n"
            f"Reference passage:\n{q['reference']}\n\nTranscript:\n\n{transcript(messages)}")


def judge(backend: Backend, model: str, q: dict, messages: list[dict]) -> dict | None:
    return backend.complete(Request(model=model, system=JUDGE_SYSTEM, user=judge_prompt(q, messages), schema=SCHEMA,
                                    max_tokens=16000, effort="medium", fallback=True)).data


def grader(spec: ModelSpec, backend: Backend) -> Callable[[dict, list[dict]], dict | None]:
    """A function (eval item, transcript) -> scores for one judge."""
    return lambda q, messages: judge(backend, spec.model, q, messages)


def score_all(graders: dict[str, Callable], q: dict, messages: list[dict]) -> dict:
    """Row fields for every judge: `judges` = {name: scores or {"error": ...}}; the first judge's scores
    are also flattened onto the row (the headline numbers, and the format of older result files)."""
    out: dict = {"judges": {}}
    for name, g in graders.items():
        try:
            out["judges"][name] = g(q, messages) or {"error": "judge declined"}
        except Exception as e:
            out["judges"][name] = {"error": str(e)}
    first = next(iter(out["judges"].values()), {"error": "no judge"})
    out.update(first if "error" not in first else {"error": first["error"]})
    return out


def reply_stats(messages: list[dict]) -> dict:
    """Measured without a judge: how long the replies are, how often they ask back, markdown use."""
    replies = [m["content"] for m in messages if m["role"] == "assistant"]
    return {
        "replies": len(replies),
        "reply_words": [len(r.split()) for r in replies],
        "asks_back": sum("?" in r for r in replies),
        "markdown": sum(bool(_MARKDOWN.search(r)) for r in replies),
    }


def _scores(row: dict, judge_name: str | None) -> dict:
    return row if judge_name is None else row.get("judges", {}).get(judge_name, {})


def summarize(rows: list[dict], arm_name: str, judge_name: str | None = None) -> dict:
    """Scores from one judge (`judge_name`, default: the headline judge) plus the measured stats."""
    mine = [r for r in rows if r["arm"] == arm_name]
    scored = [_scores(r, judge_name) for r in mine if "style" in _scores(r, judge_name)]
    talked = [r for r in mine if "reply_words" in r]
    words = sorted(w for r in talked for w in r["reply_words"])
    n_replies = sum(r["replies"] for r in talked)
    out = {"n": len(scored), "errors": sum("error" in r or "error" in _scores(r, judge_name) for r in mine)}
    for k in SCORES:
        out[k] = round(statistics.mean(r[k] for r in scored), 2) if scored else None
    out.update({
        "replies_per_conversation": round(n_replies / len(talked), 2) if talked else None,
        "reply_words_median": statistics.median(words) if words else None,
        "reply_words_p90": words[int(0.9 * (len(words) - 1))] if words else None,
        "asks_back_share": round(sum(r["asks_back"] for r in talked) / n_replies, 2) if n_replies else None,
        "markdown_share": round(sum(r["markdown"] for r in talked) / n_replies, 2) if n_replies else None,
    })
    return out


def agreement(rows: list[dict], a: str, b: str) -> dict:
    """How judge `b` scores compared with judge `a` on the same transcripts, per dimension."""
    pairs = [(r["judges"][a], r["judges"][b]) for r in rows
             if "style" in r.get("judges", {}).get(a, {}) and "style" in r.get("judges", {}).get(b, {})]
    out = {"n": len(pairs)}
    for k in SCORES:
        x, y = [p[0][k] for p in pairs], [p[1][k] for p in pairs]
        corr = statistics.correlation(x, y) if len(pairs) > 2 and len(set(x)) > 1 and len(set(y)) > 1 else None
        out[k] = {"mean_diff": round(statistics.mean(y) - statistics.mean(x), 2) if pairs else None,
                  "exact": round(sum(i == j for i, j in zip(x, y)) / len(pairs), 2) if pairs else None,
                  "within_1": round(sum(abs(i - j) <= 1 for i, j in zip(x, y)) / len(pairs), 2) if pairs else None,
                  "corr": round(corr, 2) if corr is not None else None}
    return out


def run_eval(questions_path: Path, arms: list[Arm], graders: dict[str, Callable], caller: Backend,
             caller_model: str, n: int = 50, turns: int = 4, out_dir: Path | None = None) -> dict:
    """`graders`: judge name -> grader(); the first is the headline judge. Returns
    {arm: summary}, where each summary also carries `by_judge` when there's more than one judge."""
    qs = [json.loads(line) for line in questions_path.open()]
    random.Random(0).shuffle(qs)
    qs = qs[:n]

    def one(args):
        arm, q = args
        row = {"arm": arm.name, "question": q["question"], "channel": q.get("channel", "voice")}
        try:
            messages = converse(arm, q, turns, lambda m, ch: caller_line(caller, caller_model, m, ch))
            row.update(conversation=messages, **reply_stats(messages))
        except Exception as e:
            return {**row, "error": str(e)}
        return {**row, **score_all(graders, q, messages)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, [(a, q) for a in arms for q in qs]))

    summary = summaries(rows, [a.name for a in arms], list(graders))
    if out_dir:
        write_results(out_dir, arms, rows, summary)
    return summary


def summaries(rows: list[dict], arm_names: list[str], judge_names: list[str]) -> dict:
    out = {}
    for arm in arm_names:
        out[arm] = summarize(rows, arm)
        if len(judge_names) > 1:
            out[arm]["by_judge"] = {j: {k: summarize(rows, arm, j)[k] for k in ("n", "errors", *SCORES)}
                                    for j in judge_names}
    if len(judge_names) > 1:
        out["_agreement"] = {f"{b} vs {judge_names[0]}": agreement(rows, judge_names[0], b) for b in judge_names[1:]}
    return out


def rejudge(rows: list[dict], graders: dict[str, Callable], references: dict[str, dict]) -> list[dict]:
    """Score saved transcripts with (more) judges. Scores already in a row are kept: older rows' flat
    scores become judge "original". `references` maps question -> eval item (for rows without one)."""
    def one(row):
        if "conversation" not in row:
            return row
        row = dict(row)
        judges = dict(row.get("judges") or ({"original": {k: row[k] for k in (*SCORES, "note") if k in row}}
                                            if "style" in row else {}))
        q = {"reference": row.get("reference") or references.get(row["question"], {}).get("reference"),
             "channel": row.get("channel", "voice")}
        if not q["reference"]:
            return {**row, "judges": judges}
        todo = {name: g for name, g in graders.items() if name not in judges}
        judges.update(score_all(todo, q, row["conversation"])["judges"] if todo else {})
        return {**row, "judges": judges}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, rows))


def results_stem(arms: list[Arm], when: str | None = None) -> str:
    """One file pair per run, e.g. 20261006-1142-tuned, so runs never overwrite each other."""
    names = "-".join(re.sub(r"[^A-Za-z0-9_.]+", "_", a.name) for a in arms)
    return f"{when or time.strftime('%Y%m%d-%H%M%S')}-{names}"


def write_results(out_dir: Path, arms: list[Arm], rows: list[dict], summary: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = results_stem(arms)
    (out_dir / f"{stem}.rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir / f"{stem}.rows.jsonl"

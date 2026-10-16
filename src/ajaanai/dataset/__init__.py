"""Dataset pipeline: catalog -> passages -> LLM-scripted conversations -> multi-turn chat JSONL."""

from __future__ import annotations

import json
import logging
import random
import re
from datetime import date
from pathlib import Path

from ..catalog import Catalog
from ..config import Settings
from .build import BuildStats, build_examples, conversation_candidates, write_jsonl
from .chunk import Passage, is_qa, passages_for, qa_passages

log = logging.getLogger(__name__)


# Layout: data/datasets/<synth label>/ holds the synthesis and check results for one reply mode and
# synth model (switching either never reuses the other's results). Each build writes its training
# files to a dated folder inside it, named by its dataset label, e.g.
#   data/datasets/verbatim_anthropic-claude-sonnet-5-5/verbatim_anthropic-claude-sonnet-5-5_20261014/
# The label is also the dataset's folder on the Modal volume (/datasets/<label>) and is recorded in
# its card.json and in the run.json of every model trained on it.


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-")


def synth_label(settings: Settings) -> str:
    """<reply mode>_<provider>-<synth model>, e.g. verbatim_anthropic-claude-sonnet-5-5."""
    from ..llm import parse_spec

    spec = parse_spec(settings.synth_model)
    return f"{settings.dataset_reply_mode}_{_slug(f'{spec.provider}-{spec.model}')}"


def synth_dir(settings: Settings) -> Path:
    return settings.data_dir / "datasets" / synth_label(settings)


def _build_key(label: str) -> tuple[str, int]:
    """Sort key for <synth label>_<YYYYMMDD>[-N] (a second build on one day gets -2, ...)."""
    m = re.search(r"_(\d{8})(?:-(\d+))?$", label)
    return (m.group(1), int(m.group(2) or 1)) if m else ("", 0)


def new_label(settings: Settings, today: date | None = None) -> str:
    base = f"{synth_label(settings)}_{(today or date.today()):%Y%m%d}"
    root, label, n = synth_dir(settings), base, 1
    while (root / label).exists():
        n += 1
        label = f"{base}-{n}"
    return label


def dataset_dir(settings: Settings, label: str | None = None) -> Path:
    """A built dataset's folder: the one named `label`, else the latest build for the current reply
    mode and synth model."""
    if label:
        found = sorted((settings.data_dir / "datasets").glob(f"*/{label}"))
        if not found:
            raise SystemExit(f"No dataset {label!r} under {settings.data_dir / 'datasets'}")
        return found[0]
    builds = [d for d in synth_dir(settings).glob(f"{synth_label(settings)}_*") if (d / "train.jsonl").exists()]
    if not builds:
        raise SystemExit(f"No dataset built for {synth_label(settings)} yet: run `ajaanai build-dataset`")
    return max(builds, key=lambda d: _build_key(d.name))


def style_passages(cat: Catalog) -> list[Passage]:
    """Passages in his own words (author=thanissaro), skipping canon anthologies. Retreat Q&A
    sections become one passage per real exchange (kind="qa")."""
    out: list[Passage] = []
    for doc in cat.iter_with_text(authors=["thanissaro"]):
        if doc.meta.get("anthology"):
            continue
        if is_qa(doc):
            out += [p for p in qa_passages(doc) if p.words >= 25]
        else:
            out += [p for p in passages_for(doc) if p.words >= 120]
    return out


def _sample(passages: list[Passage], n: int, qa_share: float = 0.2) -> list[Passage]:
    """A fixed random sample that always includes some real Q&A (they're ~2% of passages)."""
    rng = random.Random(0)
    qa = [p for p in passages if p.kind == "qa"]
    rest = [p for p in passages if p.kind != "qa"]
    rng.shuffle(qa)
    rng.shuffle(rest)
    k = min(len(qa), round(n * qa_share))
    return qa[:k] + rest[: n - k]


def build_dataset(cat: Catalog, settings: Settings, *, sample: int | None = None,
                  use_batches: bool = True) -> tuple[BuildStats, Path]:
    """Synthesize (resuming from stored results), check, and write the training files. Returns the
    stats and the folder the files went to: a new labelled build, or <synth dir>/sample."""
    from ..llm import make_backend, parse_spec
    from .check import check_params
    from .synth import SynthStore, _params, input_sha, run_batch, run_sync

    root = synth_dir(settings)
    passages = style_passages(cat)
    log.info("%d style passages", len(passages))
    if sample:
        passages = _sample(passages, sample)
        root = root / "sample"
    run = run_batch if use_batches and not sample else run_sync

    # 1. The synth model scripts the conversations.
    def expected(items):
        return {i: input_sha(params) for i, params in items}

    synth_store = SynthStore(root)
    shaped = settings.dataset_reply_mode == "shaped"
    synth_spec = parse_spec(settings.synth_model)
    synth_items = [(p.id, _params(synth_spec.model, p, shaped=shaped)) for p in passages]
    run(make_backend(synth_spec, settings), synth_items, synth_store)
    synth = synth_store.load(expected(synth_items))
    extras = dict(ask_back=settings.dataset_ask_back, acknowledge=settings.dataset_acknowledge, shaped=shaped)

    # 2. A cheap model labels whether each reply answers the caller's line.
    responsive = None
    if settings.dataset_check:
        candidates = conversation_candidates(passages, synth, **extras)
        log.info("checking %d conversations for responsiveness", len(candidates))
        check_store = SynthStore(root, prefix="check")
        check_spec = parse_spec(settings.check_model)
        check_items = [(c.conv_id, check_params(check_spec.model, c.turns)) for c in candidates]
        run(make_backend(check_spec, settings), check_items, check_store)
        responsive = {cid: r.get("responsive") for cid, r in check_store.load(expected(check_items)).items()}

    # 3. Assemble the training files.
    train, evals, eval_qs, stats = build_examples(passages, synth, settings.eval_fraction, responsive=responsive,
                                                  **extras)
    label = f"{synth_label(settings)}_sample" if sample else new_label(settings)
    out = root if sample else root / label
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", evals)
    write_jsonl(out / "eval_questions.jsonl", eval_qs)
    card = {"label": label, "reply_mode": settings.dataset_reply_mode, "passages": len(passages), "train": stats.train, "eval": stats.eval,
            "by_type": stats.by_type, "by_channel": stats.by_channel, "skipped_unusable": stats.skipped_unusable,
            "assistant_turns": stats.assistant_turns, "ask_backs": stats.ask_backs,
            "acknowledgements": stats.acknowledgements, "turns_dropped": stats.turns_dropped,
            "turns_not_responsive": stats.turns_not_responsive, "synth_model": settings.synth_model,
            "check_model": settings.check_model if settings.dataset_check else None,
            "licence": "Source texts © Thanissaro Bhikkhu, CC BY-NC 4.0 (dhammatalks.org). Non-commercial use only."}
    (out / "card.json").write_text(json.dumps(card, indent=2))
    log.info("dataset: %s", card)
    return stats, out

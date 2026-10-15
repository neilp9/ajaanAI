"""Dataset pipeline: catalog -> passages -> Claude-scripted conversations -> multi-turn chat JSONL."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ..catalog import Catalog
from ..config import Settings
from .build import BuildStats, build_examples, conversation_candidates, write_jsonl
from .chunk import Passage, is_qa, passages_for, qa_passages

log = logging.getLogger(__name__)


def dataset_dir(settings: Settings) -> Path:
    """data/dataset (verbatim replies) or data/dataset-shaped: each reply mode keeps its own synthesis
    results, check labels and training files, so switching modes never reuses the other's results."""
    return settings.data_dir / ("dataset" if settings.dataset_reply_mode == "verbatim" else "dataset-shaped")


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
                  use_batches: bool = True) -> BuildStats:
    import anthropic

    from .check import check_params
    from .synth import SynthStore, _params, input_sha, run_batch, run_sync

    root = dataset_dir(settings)
    passages = style_passages(cat)
    log.info("%d style passages", len(passages))
    if sample:
        passages = _sample(passages, sample)
        root = root / "sample"
    batched = use_batches and not sample
    run = (lambda items, store: run_batch(client, items, store)) if batched else \
          (lambda items, store: run_sync(client, items, store))
    # Falls back to the SDK's default credentials (env / `ant auth login` profile) when unset.
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else anthropic.Anthropic()

    # 1. Claude scripts the conversations.
    def expected(items):
        return {i: input_sha(params) for i, params in items}

    synth_store = SynthStore(root)
    shaped = settings.dataset_reply_mode == "shaped"
    synth_items = [(p.id, _params(settings.synth_model, p, shaped=shaped)) for p in passages]
    run(synth_items, synth_store)
    synth = synth_store.load(expected(synth_items))
    extras = dict(ask_back=settings.dataset_ask_back, acknowledge=settings.dataset_acknowledge, shaped=shaped)

    # 2. A cheap model labels whether each reply answers the caller's line.
    responsive = None
    if settings.dataset_check:
        candidates = conversation_candidates(passages, synth, **extras)
        log.info("checking %d conversations for responsiveness", len(candidates))
        check_store = SynthStore(root, prefix="check")
        check_items = [(c.conv_id, check_params(settings.check_model, c.turns)) for c in candidates]
        run(check_items, check_store)
        responsive = {cid: r.get("responsive") for cid, r in check_store.load(expected(check_items)).items()}

    # 3. Assemble the training files.
    train, evals, eval_qs, stats = build_examples(passages, synth, settings.eval_fraction, responsive=responsive,
                                                  **extras)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "eval.jsonl", evals)
    write_jsonl(root / "eval_questions.jsonl", eval_qs)
    card = {"reply_mode": settings.dataset_reply_mode, "passages": len(passages), "train": stats.train, "eval": stats.eval,
            "by_type": stats.by_type, "by_channel": stats.by_channel, "skipped_unusable": stats.skipped_unusable,
            "assistant_turns": stats.assistant_turns, "ask_backs": stats.ask_backs,
            "acknowledgements": stats.acknowledgements, "turns_dropped": stats.turns_dropped,
            "turns_not_responsive": stats.turns_not_responsive, "synth_model": settings.synth_model,
            "check_model": settings.check_model if settings.dataset_check else None,
            "licence": "Source texts © Thanissaro Bhikkhu, CC BY-NC 4.0 (dhammatalks.org). Non-commercial use only."}
    (root / "card.json").write_text(json.dumps(card, indent=2))
    log.info("dataset: %s", card)
    return stats

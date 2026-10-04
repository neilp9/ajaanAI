"""Dataset pipeline: catalog -> passages -> Claude questions -> chat JSONL."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ..catalog import Catalog
from ..config import Settings
from .build import BuildStats, build_examples, write_jsonl
from .chunk import Passage, passages_for

log = logging.getLogger(__name__)


def dataset_dir(settings: Settings) -> Path:
    return settings.data_dir / "dataset"


def style_passages(cat: Catalog) -> list[Passage]:
    """Passages in his own words (author=thanissaro), skipping canon anthologies."""
    out: list[Passage] = []
    for doc in cat.iter_with_text(authors=["thanissaro"]):
        if doc.meta.get("anthology"):
            continue
        out += [p for p in passages_for(doc) if p.words >= 120]
    return out


def build_dataset(cat: Catalog, settings: Settings, *, sample: int | None = None,
                  use_batches: bool = True) -> BuildStats:
    import anthropic

    from .synth import SynthStore, synth_batch, synth_sync

    root = dataset_dir(settings)
    passages = style_passages(cat)
    log.info("%d style passages", len(passages))
    if sample:
        random.Random(0).shuffle(passages)
        passages = passages[:sample]
        root = root / "sample"
    store = SynthStore(root)
    # Falls back to the SDK's default credentials (env / `ant auth login` profile) when unset.
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else anthropic.Anthropic()
    if sample or not use_batches:
        done = store.done_ids()
        synth_sync(client, settings.synth_model, [p for p in passages if p.id not in done], store)
    else:
        synth_batch(client, settings.synth_model, passages, store)

    train, evals, eval_qs, stats = build_examples(passages, store.load(), settings.eval_fraction)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "eval.jsonl", evals)
    write_jsonl(root / "eval_questions.jsonl", eval_qs)
    card = {"passages": len(passages), "train": stats.train, "eval": stats.eval,
            "by_type": stats.by_type, "skipped_unusable": stats.skipped_unusable,
            "synth_model": settings.synth_model,
            "licence": "Source texts © Thanissaro Bhikkhu, CC BY-NC 4.0 (dhammatalks.org). Non-commercial use only."}
    (root / "card.json").write_text(json.dumps(card, indent=2))
    log.info("dataset: %s", card)
    return stats

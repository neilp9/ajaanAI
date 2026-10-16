"""LoRA fine-tune of an open-weights model (runs inside a Modal GPU container).

Model-agnostic: the base weights, LoRA target modules, GPU and chat-turn markers come from the
active ModelProfile. Data is the provider-neutral multi-turn chat JSONL from `ajaanai build-dataset`.
Loss is only computed on the teacher's words:
  - profiles with `turn_markers`: each conversation is rendered once with the model's chat template
    and everything but the assistant turns is masked (Unsloth's train_on_responses_only);
  - otherwise: each teacher turn becomes its own prompt/completion example (the call so far -> his
    next reply). Same objective, but conversations are repeated once per turn — several times slower.

Output: /data/models/<run_id>/{adapter,merged} + run.json. Serving uses a run only after
`ajaanai promote <run_id>` (normally after `ajaanai eval`). A `max_steps` smoke run saves the
adapter only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..models import ModelProfile

MAX_EVAL_CONVERSATIONS = 200  # enough for a loss curve; the full eval set took ~7 min per pass


def to_prompt_completions(batch: dict) -> dict:
    """Batched `datasets.map`: one example per assistant turn, with the conversation so far as prompt."""
    out: dict[str, list] = {"prompt": [], "completion": []}
    for msgs in batch["messages"]:
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                out["prompt"].append(msgs[:i])
                out["completion"].append([m])
    return out


def to_text(batch: dict, tokenizer) -> dict:
    """Batched `datasets.map`: each whole conversation rendered with the model's chat template."""
    return {"text": [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in batch["messages"]]}


def run_finetune(profile: ModelProfile, train_path: Path, eval_path: Path, models_dir: Path, *,
                 rank: int = 32, epochs: int = 2, max_seq_len: int = 4096, lr: float = 1e-4,
                 max_steps: int | None = None, hf_token: str | None = None,
                 push_repo: str | None = None, dataset: str | None = None) -> Path:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    try:  # Unsloth's generic loader handles text and multimodal checkpoints alike
        from unsloth import FastModel as Loader
    except ImportError:
        from unsloth import FastLanguageModel as Loader

    smoke = max_steps is not None
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + profile.name + ("-smoke" if smoke else "")
    out = models_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = Loader.from_pretrained(
        profile.hf_id, max_seq_length=max_seq_len, load_in_4bit=False, token=hf_token,
    )
    model = Loader.get_peft_model(
        model, r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=profile.lora_targets,
        use_gradient_checkpointing="unsloth", random_state=0,
    )
    # Multimodal checkpoints (e.g. Qwen3.5) load a processor; text training wants its tokenizer.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    files = {"train": str(train_path)}
    if eval_path.exists() and eval_path.stat().st_size:
        files["eval"] = str(eval_path)
    ds = load_dataset("json", data_files=files)
    if "eval" in ds and len(ds["eval"]) > MAX_EVAL_CONVERSATIONS:
        ds["eval"] = ds["eval"].shuffle(seed=0).select(range(MAX_EVAL_CONVERSATIONS))
    whole = profile.turn_markers is not None
    if whole:
        ds = ds.map(to_text, batched=True, remove_columns=["messages"], fn_kwargs={"tokenizer": text_tokenizer})
    else:
        ds = ds.map(to_prompt_completions, batched=True, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        processing_class=text_tokenizer if whole else tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        args=SFTConfig(
            output_dir=str(out / "checkpoints"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,  # the default (8) ran out of memory on a 27B model
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            max_steps=max_steps or -1,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="steps" if "eval" in files else "no",
            eval_steps=10 if smoke else 250,
            save_steps=500,
            save_total_limit=2,
            bf16=True,
            max_length=max_seq_len,
            **({"dataset_text_field": "text"} if whole else {"completion_only_loss": True}),
            report_to="none",
            seed=0,
        ),
    )
    if whole:
        from unsloth.chat_templates import train_on_responses_only

        instruction_part, response_part = profile.turn_markers
        trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)
    result = trainer.train()

    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    if not smoke:
        model.save_pretrained_merged(str(out / "merged"), tokenizer, save_method="merged_16bit")
        if push_repo:
            model.push_to_hub_merged(push_repo, tokenizer, save_method="merged_16bit", token=hf_token)

    (out / "run.json").write_text(json.dumps({
        "run_id": run_id, "dataset": dataset, "profile": profile.name, "base": profile.hf_id, "rank": rank, "epochs": epochs,
        "lr": lr, "train_loss": result.training_loss, "push_repo": push_repo, "smoke": smoke,
        "max_steps": max_steps, "loss_on": "whole conversations" if whole else "per-turn examples",
        "eval_history": [h for h in trainer.state.log_history if "eval_loss" in h],
    }, indent=2))
    return out


def promote(models_dir: Path, run_id: str) -> Path:
    merged = models_dir / run_id / "merged"
    if not merged.exists():
        raise SystemExit(f"No merged model at {merged}")
    (models_dir / "CURRENT").write_text(run_id)
    return merged


def current_model_path(models_dir: Path) -> Path | None:
    pointer = models_dir / "CURRENT"
    if pointer.exists():
        path = models_dir / pointer.read_text().strip() / "merged"
        if path.exists():
            return path
    return None

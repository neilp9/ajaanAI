"""LoRA fine-tune of an open-weights model (runs inside a Modal GPU container).

Model-agnostic: the base weights, LoRA target modules and GPU come from the active
ModelProfile. Data is the provider-neutral multi-turn chat JSONL from `ajaanai build-dataset`;
each conversation is expanded into one TRL prompt/completion example per teacher turn (the call so
far -> his next reply), so loss is only computed on the teacher's words, whatever the model's chat
template.

Output: /data/models/<run_id>/{adapter,merged} + run.json. Serving uses a run only after
`ajaanai promote <run_id>` (normally after `ajaanai eval`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..models import ModelProfile


def to_prompt_completions(batch: dict) -> dict:
    """Batched `datasets.map`: one example per assistant turn, with the conversation so far as prompt."""
    out: dict[str, list] = {"prompt": [], "completion": []}
    for msgs in batch["messages"]:
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                out["prompt"].append(msgs[:i])
                out["completion"].append([m])
    return out


def run_finetune(profile: ModelProfile, train_path: Path, eval_path: Path, models_dir: Path, *,
                 rank: int = 32, epochs: int = 2, max_seq_len: int = 4096, lr: float = 1e-4,
                 hf_token: str | None = None, push_repo: str | None = None) -> Path:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    try:  # Unsloth's generic loader handles text and multimodal checkpoints alike
        from unsloth import FastModel as Loader
    except ImportError:
        from unsloth import FastLanguageModel as Loader

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + profile.name
    out = models_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = Loader.from_pretrained(
        profile.hf_id, max_seq_length=max_seq_len, load_in_4bit=False, token=hf_token,
    )
    model = Loader.get_peft_model(
        model, r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=profile.lora_targets,
        use_gradient_checkpointing="unsloth", random_state=0,
    )

    files = {"train": str(train_path)}
    if eval_path.exists() and eval_path.stat().st_size:
        files["eval"] = str(eval_path)
    ds = load_dataset("json", data_files=files).map(to_prompt_completions, batched=True,
                                                    remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        args=SFTConfig(
            output_dir=str(out / "checkpoints"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="steps" if "eval" in files else "no",
            eval_steps=200,
            save_steps=500,
            save_total_limit=2,
            bf16=True,
            max_length=max_seq_len,
            completion_only_loss=True,
            report_to="none",
            seed=0,
        ),
    )
    result = trainer.train()

    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    model.save_pretrained_merged(str(out / "merged"), tokenizer, save_method="merged_16bit")
    if push_repo:
        model.push_to_hub_merged(push_repo, tokenizer, save_method="merged_16bit", token=hf_token)

    (out / "run.json").write_text(json.dumps({
        "run_id": run_id, "profile": profile.name, "base": profile.hf_id, "rank": rank, "epochs": epochs,
        "lr": lr, "train_loss": result.training_loss, "push_repo": push_repo,
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

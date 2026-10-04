"""Hosted fine-tuning backends (TRAIN_BACKEND=openai | together), using the same JSONL.

After a job finishes, point the proxy at the result with LLM_BACKEND=openai_compatible and the
printed UPSTREAM_* values.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)
T = httpx.Timeout(300.0, connect=20.0)


def _ok(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        raise SystemExit(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def openai_finetune(api_key: str, base_model: str, train: Path, eval_: Path | None, epochs: int,
                    poll_s: int = 60) -> dict:
    h = {"Authorization": f"Bearer {api_key}"}
    base = "https://api.openai.com/v1"

    def upload(p: Path) -> str:
        with p.open("rb") as f:
            return _ok(httpx.post(f"{base}/files", headers=h, data={"purpose": "fine-tune"},
                                  files={"file": (p.name, f)}, timeout=T))["id"]

    body = {"model": base_model, "training_file": upload(train), "suffix": "ajaan-geoff",
            "method": {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": epochs}}}}
    if eval_ and eval_.exists() and eval_.stat().st_size:
        body["validation_file"] = upload(eval_)
    job = _ok(httpx.post(f"{base}/fine_tuning/jobs", headers=h, json=body, timeout=T))
    log.info("openai job %s", job["id"])
    while job["status"] not in ("succeeded", "failed", "cancelled"):
        time.sleep(poll_s)
        job = _ok(httpx.get(f"{base}/fine_tuning/jobs/{job['id']}", headers=h, timeout=T))
        log.info("openai job %s: %s", job["id"], job["status"])
    if job["status"] != "succeeded":
        raise SystemExit(f"OpenAI fine-tune {job['status']}: {job.get('error')}")
    return {"LLM_BACKEND": "openai_compatible", "UPSTREAM_BASE_URL": base,
            "UPSTREAM_MODEL": job["fine_tuned_model"], "UPSTREAM_API_KEY": "<your OPENAI_API_KEY>",
            "MODEL_PROFILE": "generic", "MODEL_ID": job["fine_tuned_model"]}


def together_finetune(api_key: str, base_model: str, train: Path, eval_: Path | None, epochs: int,
                      lora_rank: int, poll_s: int = 60) -> dict:
    h = {"Authorization": f"Bearer {api_key}"}
    base = "https://api.together.xyz/v1"

    def upload(p: Path) -> str:
        with p.open("rb") as f:
            return _ok(httpx.post(f"{base}/files/upload", headers=h,
                                  data={"purpose": "fine-tune", "file_name": p.name},
                                  files={"file": (p.name, f)}, timeout=T))["id"]

    body = {"model": base_model, "training_file": upload(train), "n_epochs": epochs, "lora": True,
            "lora_r": lora_rank, "lora_alpha": lora_rank, "suffix": "ajaan-geoff"}
    if eval_ and eval_.exists() and eval_.stat().st_size:
        body["validation_file"] = upload(eval_)
        body["n_evals"] = 4
    job = _ok(httpx.post(f"{base}/fine-tunes", headers=h, json=body, timeout=T))
    log.info("together job %s", job["id"])
    while job.get("status") not in ("completed", "error", "cancelled", "user_error"):
        time.sleep(poll_s)
        job = _ok(httpx.get(f"{base}/fine-tunes/{job['id']}", headers=h, timeout=T))
        log.info("together job %s: %s", job["id"], job.get("status"))
    if job["status"] != "completed":
        raise SystemExit(f"Together fine-tune {job['status']}")
    return {"LLM_BACKEND": "openai_compatible", "UPSTREAM_BASE_URL": base,
            "UPSTREAM_MODEL": job.get("output_name") or job.get("model_output_name"),
            "UPSTREAM_API_KEY": "<your TOGETHER_API_KEY>",
            "NOTE": "Together serves fine-tunes on a dedicated endpoint — start one in their console first.",
            "MODEL_PROFILE": "generic", "MODEL_ID": base_model}

"""Everything that runs in the cloud, as one Modal app named "ajaanai".

  modal deploy src/ajaanai/modal_app.py      (or: ajaanai deploy)

Functions
  evening_job            cron (EVENING_CRON): new talks, transcript re-checks, ASR, RAG refresh
  evening_backfill       one-off full crawl of all years
  collect_job            run sources.yaml sources by name
  transcribe_job         transcribe talks without text in the shared catalog
  sources_scheduler      hourly: runs sources whose `refresh:` cron is due
  whisper_transcribe     self-hosted ASR (TRANSCRIBE_PROVIDER=modal_whisper)
  rag_update             embed new/changed passages into the retrieval index
  finetune               LoRA fine-tune of the active model profile
  Serve                  GPU: vLLM + the RAG/filler proxy (LLM_BACKEND=modal_vllm)
  proxy                  CPU: the proxy in front of a hosted model (LLM_BACKEND=openai_compatible)
  elevenlabs_init        CPU: instant conversation-start webhook that wakes the GPU in the background

All state lives on the "ajaanai-data" volume (/data); secrets come from the "ajaanai" Modal
secret (`ajaanai secrets-sync` uploads your .env).
"""

from __future__ import annotations

import logging
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import modal

from ajaanai.config import get_settings
from ajaanai.models import active_profile

ROOT = Path(__file__).resolve().parents[2]
S = get_settings()
PROFILE = active_profile()
DATA = "/data"
VLLM_PORT = 8000
SERVED_NAME = "ajaan"

app = modal.App("ajaanai")
volume = modal.Volume.from_name("ajaanai-data", create_if_missing=True)
locks = modal.Dict.from_name("ajaanai-locks", create_if_missing=True)
secret = modal.Secret.from_name("ajaanai")
ENV = {"DATA_DIR": DATA, "SOURCES_FILE": "/root/sources.yaml", "HF_HOME": f"{DATA}/hf-cache",
       "PYTHONUNBUFFERED": "1"}

CORE = ["httpx>=0.27", "selectolax>=0.3.21", "pydantic>=2.7", "pydantic-settings>=2.3", "pyyaml>=6.0",
        "typer>=0.12", "anthropic>=0.40", "numpy>=1.26", "fastapi>=0.110"]


def _finish(image: modal.Image) -> modal.Image:
    return (image.env(ENV)
            .add_local_file(ROOT / "sources.yaml", "/root/sources.yaml")
            .add_local_python_source("ajaanai"))


base_image = _finish(modal.Image.debian_slim(python_version="3.12").uv_pip_install(*CORE))
embed_image = _finish(modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    *CORE, "sentence-transformers==6.1.0"))
whisper_image = _finish(
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg").uv_pip_install(*CORE, "faster-whisper==1.2.1"))
train_image = _finish(modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    *CORE, "unsloth==2026.9.14", "trl>=0.20", "datasets"))  # unsloth bounds trl (<=0.24) and datasets
vllm_image = _finish(modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    *CORE, "vllm==0.30.0", "sentence-transformers==6.1.0").env({
        "VLLM_SERVER_DEV_MODE": "1",  # exposes /sleep and /wake_up (used around the snapshot)
        "TORCHINDUCTOR_COMPILE_THREADS": "1",  # recommended by Modal for snapshot compatibility
    }))

log = logging.getLogger("ajaanai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@contextmanager
def _catalog_writer(owner: str):
    """Hold the single-writer lock; yields a checkpoint() that commits progress to the volume."""
    from ajaanai.lock import CatalogLock

    lock = CatalogLock(locks, owner)
    with lock.held():
        volume.reload()  # start from the latest committed catalog

        def checkpoint():
            volume.commit()
            lock.heartbeat()

        try:
            yield checkpoint
        finally:
            volume.commit()


def _ctx():
    from ajaanai.catalog import open_catalog
    from ajaanai.scrape.http import PoliteClient

    get_settings.cache_clear()
    return open_catalog(), PoliteClient(), get_settings()


# --- data collection --------------------------------------------------------------------------


@app.function(image=base_image, volumes={DATA: volume}, secrets=[secret], timeout=6 * 3600,
              schedule=modal.Cron(S.evening_cron))
def evening_job():
    from ajaanai.lock import CatalogBusy
    from ajaanai.scrape.evening import run_evening

    try:
        with _catalog_writer("evening_job") as checkpoint:
            cat, http, s = _ctx()
            try:
                report = run_evening(cat, http, s, checkpoint=checkpoint)
            finally:
                cat.close()
    except CatalogBusy as e:  # e.g. a backfill is running; it covers today's talks too
        log.warning("skipping daily run: %s", e)
        return f"skipped: {e}"
    if report.changed_ids or report.discovered.site_transcripts:
        rag_update.remote()
    return report.summary()


@app.function(image=base_image, volumes={DATA: volume}, secrets=[secret], timeout=24 * 3600)
def evening_backfill(years: str | None = None, limit: int | None = None, transcribe: bool = False):
    from ajaanai.scrape.evening import run_evening

    with _catalog_writer("evening_backfill") as checkpoint:
        cat, http, s = _ctx()
        try:
            report = run_evening(cat, http, s, backfill=True, years=years, limit=limit, transcribe=transcribe,
                                 checkpoint=checkpoint)
        finally:
            cat.close()
    return report.summary()


@app.function(image=base_image, volumes={DATA: volume}, secrets=[secret], timeout=24 * 3600)
def transcribe_job(provider: str | None = None, model: str | None = None, limit: int | None = None,
                   sources: list[str] | None = None, grace_days: int | None = None,
                   retry_failed: bool = False) -> dict:
    """Transcribe talks without text in the shared catalog (`ajaanai transcribe`)."""
    from ajaanai.transcribe import transcribe_pending

    with _catalog_writer("transcribe_job") as checkpoint:
        cat, _, s = _ctx()
        try:
            ok, failed = transcribe_pending(cat, s, sources=sources, grace_days=grace_days, limit=limit,
                                            provider=provider, model=model, retry_failed=retry_failed,
                                            checkpoint=checkpoint)
        finally:
            cat.close()
    if ok:
        rag_update.spawn()
    return {"transcribed": len(ok), "failed": len(failed), "provider": provider or s.transcribe_provider}


@app.function(image=base_image, volumes={DATA: volume}, secrets=[secret], timeout=24 * 3600)
def collect_job(names: list[str] | None = None, limit: int | None = None, dry_run: bool = False):
    from ajaanai.scrape.collections import collect, load_sources

    with _catalog_writer(f"collect {' '.join(names or ['enabled'])}") as checkpoint:
        cat, http, s = _ctx()
        try:
            results = collect(load_sources(s.sources_file), names, cat, http, dry_run=dry_run, limit=limit,
                              checkpoint=checkpoint)
        finally:
            cat.close()
    return [{"source": r.name, "planned": r.planned[:50], "new": r.stats.new, "seen": r.stats.seen,
             "errors": r.stats.errors[:20]} for r in results]


@app.function(image=base_image, volumes={DATA: volume}, secrets=[secret], timeout=12 * 3600,
              schedule=modal.Cron("5 * * * *"))
def sources_scheduler():
    """Hourly: run every sources.yaml source whose `refresh` cron matches this hour."""
    from datetime import datetime, timezone

    from ajaanai.scrape.collections import due_sources, load_sources

    get_settings.cache_clear()
    due = due_sources(load_sources(get_settings().sources_file), datetime.now(timezone.utc))
    return collect_job.local(due) if due else []


@app.function(image=whisper_image, gpu="L4", volumes={DATA: volume}, timeout=3600, max_containers=10)
def whisper_transcribe(audio_url: str, model: str, glossary: list[str]) -> str:
    import tempfile

    import httpx
    from faster_whisper import WhisperModel

    from ajaanai.transcribe.base import paragraphize

    global _WHISPER
    if "_WHISPER" not in globals() or _WHISPER[0] != model:
        _WHISPER = (model, WhisperModel(model, device="cuda", compute_type="float16",
                                        download_root=f"{DATA}/hf-cache/whisper"))
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        with httpx.stream("GET", audio_url, timeout=600, follow_redirects=True) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                f.write(chunk)
        f.flush()
        segments, _ = _WHISPER[1].transcribe(
            f.name, language="en", vad_filter=True, beam_size=5,
            initial_prompt="A Dhamma talk by Thanissaro Bhikkhu. " + ", ".join(glossary[:40]))
        return paragraphize(" ".join(s.text.strip() for s in segments))


@app.function(image=embed_image, gpu="L4", volumes={DATA: volume}, secrets=[secret], timeout=6 * 3600,
              max_containers=1)  # one index writer at a time; later calls queue
def rag_update() -> int:
    from ajaanai.catalog import open_catalog
    from ajaanai.rag.index import Embedder, RagIndex

    volume.reload()
    get_settings.cache_clear()
    s = get_settings()
    idx = RagIndex(s.data_dir / "rag")
    added = idx.update(open_catalog(), Embedder(s.embed_model, device="cuda"))
    volume.commit()
    return added


# --- training ------------------------------------------------------------------------------


@app.function(image=train_image, gpu=PROFILE.train_gpu, volumes={DATA: volume}, secrets=[secret],
              timeout=24 * 3600)
def finetune(dataset_subdir: str = "dataset") -> str:
    from ajaanai.train.finetune import run_finetune

    volume.reload()
    get_settings.cache_clear()
    s = get_settings()
    root = s.data_dir / dataset_subdir
    out = run_finetune(active_profile(), root / "train.jsonl", root / "eval.jsonl", s.data_dir / "models",
                       rank=s.lora_rank, epochs=s.train_epochs, hf_token=s.hf_token, push_repo=s.hf_output_repo)
    volume.commit()
    return out.name


# --- serving --------------------------------------------------------------------------------


def _proxy_config(upstream_base: str, upstream_model: str, upstream_key: str | None, hosted: bool):
    from ajaanai.serve.proxy import ProxyConfig

    from ajaanai.dataset.persona import FILLERS_FIRST, FILLERS_REPEAT

    s = get_settings()
    s.require("endpoint_bearer_secret")  # never expose an unauthenticated GPU endpoint

    return ProxyConfig(
        upstream_base_url=upstream_base, upstream_model=upstream_model, upstream_api_key=upstream_key,
        profile=active_profile(), reasoning_mode=s.reasoning_mode,
        reasoning_budget_tokens=s.reasoning_budget_tokens, max_answer_tokens=s.max_answer_tokens,
        filler_mode=s.filler_mode, filler_interval_s=s.filler_interval_s,
        fillers_first=tuple(s.fillers_first or FILLERS_FIRST),
        fillers_repeat=tuple(s.fillers_repeat or FILLERS_REPEAT),
        send_template_kwargs=not hosted, bearer_secret=s.endpoint_bearer_secret,
    )


def _embedder(device: str):
    from ajaanai.rag.index import Embedder

    s = get_settings()
    return Embedder(s.embed_model, device=device) if s.rag_enabled else None


def _retriever(embedder):
    """Loads the retrieval index fresh (it changes daily), around an already-loaded embedder."""
    from ajaanai.rag.index import RagIndex, Retriever, format_hits

    s = get_settings()
    idx = RagIndex(s.data_dir / "rag")
    if embedder is None or len(idx) == 0:
        return None
    r = Retriever(idx, embedder, s.rag_top_k)
    return lambda q: format_hits(r(q))


def _wait_vllm_healthy(proc: subprocess.Popen, timeout_s: float = 1700) -> None:
    import httpx

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("vLLM did not become healthy")


def _vllm_post(path: str, json: dict | None = None) -> None:
    import httpx

    httpx.post(f"http://127.0.0.1:{VLLM_PORT}{path}", json=json, timeout=300).raise_for_status()


SNAPSHOT = S.serve_gpu_snapshot


@app.cls(image=vllm_image, gpu=PROFILE.serve_gpu, volumes={DATA: volume}, secrets=[secret],
         scaledown_window=S.serve_scaledown_window_s, min_containers=S.serve_min_containers,
         timeout=3600, startup_timeout=1800, enable_memory_snapshot=SNAPSHOT,
         experimental_options={"enable_gpu_snapshot": True} if SNAPSHOT else None)
@modal.concurrent(max_inputs=32)
class Serve:
    """vLLM + proxy. With SERVE_GPU_SNAPSHOT, `boot` runs once per deploy and is snapshotted;
    cold starts then only run `resume` (wake vLLM, load today's retrieval index)."""

    @modal.enter(snap=SNAPSHOT)
    def boot(self):
        from ajaanai.train.finetune import current_model_path

        get_settings.cache_clear()
        s = get_settings()
        profile = active_profile()
        model = str(current_model_path(s.data_dir / "models") or profile.hf_id)
        log.info("booting vLLM for %s (%s), snapshot=%s", model, profile.name, SNAPSHOT)
        self.proc = subprocess.Popen(profile.vllm_args(model, SERVED_NAME, VLLM_PORT, sleep_mode=SNAPSHOT,
                                                       max_num_seqs=s.serve_max_num_seqs))
        _wait_vllm_healthy(self.proc)
        if SNAPSHOT:
            # Exercise both reasoning paths so compiled kernels land in the snapshot, then sleep:
            # weights move to CPU memory and the KV cache is dropped, keeping the snapshot lean.
            for think in (False, True, False):
                _vllm_post("/v1/chat/completions", {
                    "model": SERVED_NAME, "max_tokens": 16,
                    "messages": [{"role": "user", "content": "How should I begin meditating?"}],
                    **({"chat_template_kwargs": profile.chat_template_kwargs(think)}
                       if profile.chat_template_kwargs(think) else {}),
                })
            _vllm_post("/sleep?level=1")
        self.embedder = _embedder("cuda")  # static model: fine to keep in the snapshot

    @modal.enter()
    def resume(self):
        if SNAPSHOT:
            _vllm_post("/wake_up")
            _wait_vllm_healthy(self.proc, timeout_s=600)
        try:  # see the latest retrieval index, not the one from snapshot time
            volume.reload()
        except Exception as e:
            log.warning("volume reload failed (%s); using the mounted index", e)
        get_settings.cache_clear()
        self.retrieve = _retriever(self.embedder)

    @modal.method()
    def warm(self) -> bool:
        return True

    @modal.asgi_app()
    def web(self):
        from ajaanai.serve.proxy import create_app

        cfg = _proxy_config(f"http://127.0.0.1:{VLLM_PORT}/v1", SERVED_NAME, None, hosted=False)
        return create_app(cfg, retrieve=self.retrieve)

    @modal.exit()
    def stop(self):
        self.proc.terminate()


@app.function(image=embed_image, volumes={DATA: volume}, secrets=[secret], min_containers=0,
              scaledown_window=600, timeout=3600)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def proxy():
    """Proxy for LLM_BACKEND=openai_compatible (OpenAI/Together fine-tunes, any hosted model)."""
    from ajaanai.serve.proxy import create_app

    get_settings.cache_clear()
    s = get_settings()
    s.require("upstream_base_url", "upstream_model")
    cfg = _proxy_config(s.upstream_base_url, s.upstream_model, s.upstream_api_key, hosted=True)
    return create_app(cfg, retrieve=_retriever(_embedder("cpu")))


@app.function(image=base_image, secrets=[secret])
@modal.fastapi_endpoint(method="POST")
def elevenlabs_init():
    """Called by ElevenLabs as a call connects. Returns immediately; wakes the GPU in the background."""
    if get_settings().llm_backend == "modal_vllm":
        Serve().warm.spawn()
    return {"type": "conversation_initiation_client_data", "dynamic_variables": {}}

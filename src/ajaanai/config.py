"""All runtime settings, read from environment variables / `.env`.

Every knob in the system lives here so it can be changed without touching code.
On Modal the same variables are injected from the `ajaanai` secret (see `ajaanai secrets-sync`).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

TranscribeProvider = Literal["elevenlabs", "openai", "deepgram", "assemblyai", "modal_whisper"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- storage -------------------------------------------------------------
    data_dir: Path = Path("data")  # set to /data on Modal (the mounted volume)

    # --- scraping --------------------------------------------------------------
    site_base_url: str = "https://www.dhammatalks.org"
    user_agent: str = "ajaanAI-research-bot/0.2 (+non-commercial; contact: see repo)"
    request_delay_s: float = 1.0
    evening_cron: str = "0 9 * * *"  # daily, UTC (talks are usually posted ~02:00-03:00 UTC)
    recheck_days: int = 120  # keep re-checking untranscribed talks this long for an official transcript
    transcribe_grace_days: int = 21  # wait this long for an official transcript before paying for ASR
    sources_file: Path = Path("sources.yaml")

    # --- transcription -------------------------------------------------------
    transcribe_provider: TranscribeProvider = "elevenlabs"
    transcribe_model: str | None = None  # provider default when unset
    transcribe_concurrency: int = 4

    # --- API keys (only the ones you use need to be set) ---------------------
    elevenlabs_api_key: str | None = None
    openai_api_key: str | None = None
    deepgram_api_key: str | None = None
    assemblyai_api_key: str | None = None
    together_api_key: str | None = None
    anthropic_api_key: str | None = None
    hf_token: str | None = None
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    endpoint_bearer_secret: str | None = None

    # --- dataset synthesis -----------------------------------------------------
    synth_model: str = "claude-haiku-4-5"  # writes the training conversations (Batches API at 50% cost)
    judge_model: str = "claude-opus-5-5"  # eval judge
    eval_fraction: float = 0.02
    # Keep the short questions back to the caller that Claude adds to some of his turns (the only
    # assistant words that aren't his). False rebuilds the dataset without them; no re-synthesis.
    dataset_ask_back: bool = True

    # --- model (swappable; see models.py) --------------------------------------
    model_profile: str = "qwen3.5-27b"  # key in models.PROFILES, or "generic"
    model_id: str | None = None  # override the profile's HF weights (required for "generic")
    model_thinking: Literal["none", "template_kwarg", "always"] | None = None
    train_gpu: str | None = None  # override the profile's GPU choices
    serve_gpu: str | None = None

    # --- training ----------------------------------------------------------------
    # modal: LoRA on our own GPU (open weights)  | openai / together: hosted fine-tuning APIs
    train_backend: Literal["modal", "openai", "together"] = "modal"
    hosted_base_model: str | None = None  # base model name for openai/together, e.g. "gpt-4.1-mini-2025-04-14"
    hf_output_repo: str | None = None  # optional private HF repo to push merged weights to
    lora_rank: int = 32
    train_epochs: int = 2

    # --- serving -------------------------------------------------------------------
    # modal_vllm: our own vLLM on Modal | openai_compatible: any OpenAI-style API (OpenAI fine-tune,
    # Together/Fireworks dedicated endpoint, local llama.cpp/Ollama, ...). The RAG + filler proxy
    # sits in front of either.
    llm_backend: Literal["modal_vllm", "openai_compatible"] = "modal_vllm"
    upstream_base_url: str | None = None  # for openai_compatible, e.g. https://api.openai.com/v1
    upstream_api_key: str | None = None
    upstream_model: str | None = None  # e.g. "ft:gpt-4.1-mini:org::abc123"
    serve_scaledown_window_s: int = 600
    # GPU memory snapshot: restore a warmed-up vLLM instead of booting it (cold start ~1-2 min ->
    # ~10-30 s). Snapshots are taken per deploy; redeploy after `promote` or serving-setting changes.
    serve_gpu_snapshot: bool = True
    serve_max_num_seqs: int = 8  # concurrent conversations per GPU; small keeps snapshots lean
    serve_min_containers: int = 0
    endpoint_url: str | None = None  # filled in after `ajaanai serve-deploy`
    rag_enabled: bool = True
    rag_top_k: int = 4
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"  # any sentence-transformers model
    max_answer_tokens: int = 400  # phone answers should be short

    # --- reasoning & pauses ------------------------------------------------------
    reasoning_mode: Literal["off", "on", "auto"] = "auto"
    reasoning_budget_tokens: int = 1024
    filler_mode: Literal["off", "when_thinking", "always"] = "when_thinking"
    filler_interval_s: float = 4.0
    fillers_first: list[str] = Field(default_factory=list)  # empty -> persona defaults
    fillers_repeat: list[str] = Field(default_factory=list)

    # --- telephony / ElevenLabs agent --------------------------------------------
    agent_name: str = "Ajaan Geoff (AI)"
    agent_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # stock ElevenLabs voice until a consented clone exists
    agent_voice_is_clone: bool = False  # true -> requires CONSENT.md to record granted permission
    agent_tts_model: str = "eleven_flash_v2"
    agent_turn_timeout_s: float = 7.0
    agent_silence_end_call_s: int = 60
    agent_max_call_s: int = 1800
    twilio_phone_number: str | None = None  # E.164, e.g. +14155550123
    consent_file: Path = Path("CONSENT.md")

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.sqlite"

    def require(self, *names: str) -> None:
        """Fail fast with a clear message when a needed key isn't configured."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            env = ", ".join(n.upper() for n in missing)
            raise SystemExit(f"Missing configuration: set {env} in .env")


@lru_cache
def get_settings() -> Settings:
    return Settings()

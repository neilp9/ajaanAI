"""Model profiles: everything model-specific lives here, so the base LLM is swappable.

Pick one with MODEL_PROFILE (a key below) and optionally override the weights with MODEL_ID.
Any Hugging Face chat model works with the `generic` profile; add a profile only when a model
needs special handling (thinking toggles, tags, LoRA targets, GPU sizing).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

ThinkingStyle = Literal[
    "none",                # model has no reasoning mode
    "template_kwarg",      # chat template takes {thinking_kwarg: bool} (Qwen3/3.5 `enable_thinking`)
    "always",              # model always reasons (we can only strip it)
]

_STD_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class ModelProfile:
    name: str
    hf_id: str
    thinking: ThinkingStyle = "none"
    thinking_kwarg: str = "enable_thinking"
    think_open: str = "<think>"
    think_close: str = "</think>"
    lora_targets: list[str] = field(default_factory=lambda: list(_STD_LORA_TARGETS))
    train_gpu: str = "H100"
    serve_gpu: str = "L40S"
    serve_quantization: str | None = "fp8"  # vLLM --quantization; None = native dtype
    max_model_len: int = 16384
    vllm_extra_args: list[str] = field(default_factory=list)
    vllm_reasoning_parser: str | None = None  # splits reasoning into `reasoning_content`
    vllm_tool_parser: str | None = None  # lets the voice platform's tools (e.g. end_call) work
    notes: str = ""

    def vllm_args(self, model_path: str, served_name: str, port: int) -> list[str]:
        args = ["vllm", "serve", model_path, "--served-model-name", served_name, "--host", "127.0.0.1",
                "--port", str(port), "--max-model-len", str(self.max_model_len),
                "--gpu-memory-utilization", "0.85"]
        if self.serve_quantization:
            args += ["--quantization", self.serve_quantization]
        if self.vllm_reasoning_parser:
            args += ["--reasoning-parser", self.vllm_reasoning_parser]
        if self.vllm_tool_parser:
            args += ["--enable-auto-tool-choice", "--tool-call-parser", self.vllm_tool_parser]
        return args + list(self.vllm_extra_args)

    def chat_template_kwargs(self, think: bool) -> dict:
        if self.thinking == "template_kwarg":
            return {self.thinking_kwarg: think}
        return {}

    @property
    def supports_thinking_toggle(self) -> bool:
        return self.thinking == "template_kwarg"


PROFILES: dict[str, ModelProfile] = {
    p.name: p
    for p in [
        ModelProfile(
            name="qwen3.5-27b", hf_id="Qwen/Qwen3.5-27B", thinking="template_kwarg", vllm_reasoning_parser="qwen3", vllm_tool_parser="hermes",
            train_gpu="H100", serve_gpu="L40S", serve_quantization="fp8",
            notes="Default. Dense 27B, Apache 2.0. Hybrid Gated-DeltaNet: needs recent vLLM/Unsloth.",
        ),
        ModelProfile(
            name="qwen3.5-9b", hf_id="Qwen/Qwen3.5-9B", thinking="template_kwarg", vllm_reasoning_parser="qwen3", vllm_tool_parser="hermes",
            train_gpu="L40S", serve_gpu="L4", serve_quantization="fp8",
            notes="Cheaper/faster; fits a 24GB GPU in FP8.",
        ),
        ModelProfile(
            name="qwen3-32b", hf_id="Qwen/Qwen3-32B", thinking="template_kwarg", vllm_reasoning_parser="qwen3", vllm_tool_parser="hermes",
            train_gpu="H100", serve_gpu="L40S", serve_quantization="fp8",
            notes="Fallback if Qwen3.5 tooling lags.",
        ),
        ModelProfile(
            name="llama-3.1-8b", hf_id="meta-llama/Llama-3.1-8B-Instruct", thinking="none", vllm_tool_parser="llama3_json",
            train_gpu="L40S", serve_gpu="L4", serve_quantization="fp8",
            notes="Llama community licence (gated on HF).",
        ),
        ModelProfile(
            name="llama-3.3-70b", hf_id="meta-llama/Llama-3.3-70B-Instruct", thinking="none", vllm_tool_parser="llama3_json",
            train_gpu="H100:2", serve_gpu="H100", serve_quantization="fp8",
        ),
        ModelProfile(
            name="generic", hf_id="", thinking="none",
            notes="Any HF chat model: set MODEL_ID (and MODEL_THINKING if it reasons).",
        ),
    ]
}


def get_profile(name: str, model_id: str | None = None, thinking: ThinkingStyle | None = None,
                train_gpu: str | None = None, serve_gpu: str | None = None) -> ModelProfile:
    if name not in PROFILES:
        raise SystemExit(f"Unknown MODEL_PROFILE {name!r}. Choose from: {', '.join(PROFILES)}")
    p = PROFILES[name]
    overrides = {k: v for k, v in dict(hf_id=model_id, thinking=thinking, train_gpu=train_gpu,
                                       serve_gpu=serve_gpu).items() if v}
    p = replace(p, **overrides)
    if not p.hf_id:
        raise SystemExit("MODEL_PROFILE=generic needs MODEL_ID=<huggingface org/name>")
    return p


def active_profile() -> ModelProfile:
    from .config import get_settings

    s = get_settings()
    return get_profile(s.model_profile, s.model_id, s.model_thinking, s.train_gpu, s.serve_gpu)

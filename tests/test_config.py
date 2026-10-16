import pytest
from pydantic import ValidationError

from ajaanai.config import Settings


def test_blank_env_values_mean_unset(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=   # optional\nENDPOINT_URL=\nSYNTH_MODEL=openai:gpt-x  # note\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("UPSTREAM_API_KEY", "")  # e.g. a blank value in the Modal secret
    s = Settings(_env_file=env)
    assert s.hf_token is None and s.endpoint_url is None and s.upstream_api_key is None
    assert s.synth_model == "openai:gpt-x"


def test_every_llm_model_names_its_provider(monkeypatch):
    for name in ("SYNTH_MODEL", "CHECK_MODEL", "JUDGE_MODEL", "EVAL_CALLER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert all(m.startswith("anthropic:") for m in (s.synth_model, s.check_model, s.judge_model, s.eval_caller_model))
    assert Settings(_env_file=None, judge_model="anthropic:a, openai:b").judge_model == "anthropic:a, openai:b"
    for bad in (dict(synth_model="claude-haiku-4-5"), dict(check_model="gpt-5-mini"),
                dict(judge_model="anthropic:a,b"), dict(eval_caller_model="gemini:x")):
        with pytest.raises(ValidationError, match="anthropic:<model> or openai:<model>"):
            Settings(_env_file=None, **bad)

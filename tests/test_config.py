from ajaanai.config import Settings


def test_blank_env_values_mean_unset(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=   # optional\nENDPOINT_URL=\nSYNTH_MODEL=claude-x  # note\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("UPSTREAM_API_KEY", "")  # e.g. a blank value in the Modal secret
    s = Settings(_env_file=env)
    assert s.hf_token is None and s.endpoint_url is None and s.upstream_api_key is None
    assert s.synth_model == "claude-x"

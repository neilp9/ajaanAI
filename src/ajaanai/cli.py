"""ajaanai command line. Run `ajaanai --help` for the list of commands.

Data lives on the Modal volume "ajaanai-data" (source of truth for scheduled jobs). Commands that
need the catalog locally (build-dataset, voice-clips) pull it first; `--local` runs scraping
against ./data instead of Modal.
"""

from __future__ import annotations

import json
import logging
import secrets as pysecrets
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from .config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
APP_NAME = "ajaanai"
VOLUME = "ajaanai-data"


def _setup_logging(verbose: bool = False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _fn(name: str):
    import modal

    return modal.Function.from_name(APP_NAME, name)


def _run_or_spawn(name: str, wait: bool, **kwargs):
    fn = _fn(name)
    if wait:
        return fn.remote(**kwargs)
    call = fn.spawn(**kwargs)
    typer.echo(f"Started {name} on Modal (call {call.object_id}). Follow it with: uv run modal app logs {APP_NAME}")
    return None


def _modal_volume(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "modal", "volume", *args], check=True)


def _pull_catalog() -> None:
    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    _modal_volume("get", "--force", VOLUME, "catalog.sqlite", str(s.catalog_path))


# --- setup / deploy -----------------------------------------------------------------------------


@app.command("secrets-sync")
def secrets_sync(env_file: Path = typer.Option(Path(".env"), help="dotenv file to upload")):
    """Upload .env to the Modal secret 'ajaanai' (creates ENDPOINT_BEARER_SECRET if missing)."""
    if not env_file.exists():
        raise typer.Exit(f"{env_file} not found — copy .env.example to .env first")
    from dotenv import dotenv_values

    if not (dotenv_values(env_file).get("ENDPOINT_BEARER_SECRET") or "").strip():
        token = pysecrets.token_urlsafe(32)
        lines = [l for l in env_file.read_text().splitlines() if not l.startswith("ENDPOINT_BEARER_SECRET=")]
        env_file.write_text("\n".join(lines).rstrip("\n") + f"\nENDPOINT_BEARER_SECRET={token}\n")
        typer.echo("Generated ENDPOINT_BEARER_SECRET in .env")
    subprocess.run([sys.executable, "-m", "modal", "secret", "create", APP_NAME, "--from-dotenv", str(env_file),
                    "--force"], check=True)


@app.command()
def deploy():
    """Deploy all cloud functions (cron jobs, GPU workers, endpoints) to Modal."""
    subprocess.run([sys.executable, "-m", "modal", "deploy", "-m", "ajaanai.modal_app"], check=True)
    endpoints()


@app.command()
def endpoints():
    """Print the deployed endpoint URLs."""
    import modal

    s = get_settings()
    try:
        if s.llm_backend == "modal_vllm":
            llm = modal.Cls.from_name(APP_NAME, "Serve")().web.get_web_url()
        else:
            llm = _fn("proxy").get_web_url()
        init = _fn("elevenlabs_init").get_web_url()
    except Exception as e:
        raise typer.Exit(f"Could not look up endpoints (deployed yet?): {e}")
    typer.echo(f"LLM endpoint (set ENDPOINT_URL): {llm}")
    typer.echo(f"ElevenLabs init webhook:          {init}")


@app.command()
def models():
    """List model profiles (MODEL_PROFILE)."""
    from .models import PROFILES, active_profile

    cur = active_profile().name
    for p in PROFILES.values():
        mark = "*" if p.name == cur else " "
        typer.echo(f"{mark} {p.name:16} {p.hf_id or '(set MODEL_ID)':38} train={p.train_gpu:7} serve={p.serve_gpu:5} "
                   f"thinking={p.thinking:14} {p.notes}")


# --- data collection ------------------------------------------------------------------------------


@app.command()
def evening(
    backfill: bool = typer.Option(False, help="Crawl every year page, not just recent talks"),
    years: Optional[str] = typer.Option(None, help="With --backfill: e.g. 2000-2010,2024"),
    limit: Optional[int] = typer.Option(None, help="Process at most N talks (testing)"),
    transcribe: Optional[bool] = typer.Option(
        None, "--transcribe/--no-transcribe",
        help="Send talks past the grace period to ASR (costs money). Default: on for the daily run, "
             "off for --backfill"),
    local: bool = typer.Option(False, help="Run here against ./data instead of on Modal"),
    wait: bool = typer.Option(False, help="Wait for the Modal job to finish"),
):
    """Component 1: evening talks (this is what the daily cron runs)."""
    _setup_logging()
    if transcribe is None:
        transcribe = not backfill  # a backfill never starts a large paid ASR run implicitly
    if local:
        from .catalog import open_catalog
        from .scrape.evening import run_evening
        from .scrape.http import PoliteClient

        r = run_evening(open_catalog(), PoliteClient(), get_settings(), backfill=backfill, years=years,
                        limit=limit, transcribe=transcribe)
        typer.echo(r.summary())
    elif backfill:
        typer.echo(_run_or_spawn("evening_backfill", wait, years=years, limit=limit, transcribe=transcribe))
    else:
        typer.echo(_run_or_spawn("evening_job", wait))


@app.command()
def collect(
    names: Optional[list[str]] = typer.Argument(None, help="Source names from sources.yaml (default: enabled)"),
    list_: bool = typer.Option(False, "--list", help="Show sources and catalog counts"),
    dry_run: bool = typer.Option(False, help="Show what would be fetched"),
    limit: Optional[int] = typer.Option(None),
    local: bool = typer.Option(False, help="Run here against ./data instead of on Modal"),
    wait: bool = typer.Option(True, help="Wait for the Modal job to finish"),
):
    """Component 2: collect other parts of the site, configured in sources.yaml."""
    _setup_logging()
    from .scrape.collections import collect as run_collect
    from .scrape.collections import load_sources

    s = get_settings()
    sources = load_sources(s.sources_file)
    if list_:
        counts = {}
        if s.catalog_path.exists():
            from .catalog import open_catalog

            for row in open_catalog().counts():
                counts.setdefault(row["source"], 0)
                counts[row["source"]] += row["n"]
        for name, spec in sources.items():
            flag = "on " if spec.enabled else "off"
            typer.echo(f"[{flag}] {name:22} {spec.kind:13} docs={counts.get(name, 0):<6} {spec.description}")
        typer.echo("(counts are from the local catalog; `ajaanai status` pulls the latest)")
        return
    if local:
        from .catalog import open_catalog
        from .scrape.http import PoliteClient

        results = run_collect(sources, names, open_catalog(), PoliteClient(), dry_run=dry_run, limit=limit)
        for r in results:
            typer.echo(f"{r.name}: new={r.stats.new} seen={r.stats.seen} errors={len(r.stats.errors)}")
            for line in r.planned:
                typer.echo(f"  {line}")
    else:
        out = _run_or_spawn("collect_job", wait, names=names, limit=limit, dry_run=dry_run)
        if out:
            typer.echo(json.dumps(out, indent=2, ensure_ascii=False))


@app.command()
def transcribe(
    provider: Optional[str] = typer.Option(None, help="Override TRANSCRIBE_PROVIDER for this run"),
    model: Optional[str] = typer.Option(None, help="Override TRANSCRIBE_MODEL"),
    limit: Optional[int] = typer.Option(None),
    source: Optional[list[str]] = typer.Option(None, help="Restrict to these sources"),
    grace_days: Optional[int] = typer.Option(None, help="Override TRANSCRIBE_GRACE_DAYS"),
    retry_failed: bool = typer.Option(False, help="Also retry talks whose transcription failed before"),
    local: bool = typer.Option(False, help="Run here against ./data instead of the shared catalog on Modal"),
    wait: bool = typer.Option(False, help="Wait for the Modal job to finish"),
):
    """Transcribe catalogued talks that have no transcript, with the configured ASR provider."""
    _setup_logging()
    kwargs = dict(provider=provider, model=model, limit=limit, sources=source, grace_days=grace_days,
                  retry_failed=retry_failed)
    if not local:
        out = _run_or_spawn("transcribe_job", wait, **kwargs)
        if out:
            typer.echo(out)
        return
    from .catalog import open_catalog
    from .transcribe import transcribe_pending

    ok, failed = transcribe_pending(open_catalog(), get_settings(), **kwargs)
    typer.echo(f"transcribed={len(ok)} failed={len(failed)}")


@app.command()
def status(pull: bool = typer.Option(True, help="Fetch the latest catalog from Modal first")):
    """Catalog counts by source / status / text source."""
    if pull:
        _pull_catalog()
    from .catalog import open_catalog

    for row in open_catalog().counts():
        typer.echo(f"{row['source']:22} {row['status']:17} {row['text_source']:22} {row['n']}")


# --- dataset / training / eval ----------------------------------------------------------------


@app.command("build-dataset")
def build_dataset_cmd(
    sample: Optional[int] = typer.Option(None, help="Synthesize for N random passages only (sync, quick look)"),
    no_batch: bool = typer.Option(False, help="Use synchronous calls instead of the Batches API"),
    pull: bool = typer.Option(True, help="Pull the catalog from Modal first"),
    push: bool = typer.Option(True, help="Upload the dataset to the Modal volume afterwards"),
):
    """Turn the catalog into chat JSONL training data (questions written by Claude, answers his words)."""
    _setup_logging()
    if pull:
        _pull_catalog()
    from .catalog import open_catalog
    from .dataset import build_dataset, dataset_dir

    s = get_settings()
    stats = build_dataset(open_catalog(), s, sample=sample, use_batches=not no_batch)
    typer.echo(f"train={stats.train} eval={stats.eval} by_type={stats.by_type}")
    if push and not sample:
        _modal_volume("put", "--force", VOLUME, str(dataset_dir(s)), "/dataset")


@app.command()
def train(
    backend: Optional[str] = typer.Option(None, help="modal | openai | together (default TRAIN_BACKEND)"),
    wait: bool = typer.Option(False, help="Wait for the Modal job"),
):
    """Fine-tune on the dataset with the chosen backend."""
    _setup_logging()
    s = get_settings()
    backend = backend or s.train_backend
    root = s.data_dir / "dataset"
    if backend == "modal":
        out = _run_or_spawn("finetune", wait)
        if out:
            typer.echo(f"Trained run {out}. Evaluate it, then: ajaanai promote {out}")
        return
    from .train import hosted

    if not s.hosted_base_model:
        raise typer.Exit("Set HOSTED_BASE_MODEL (the provider's base model name) for hosted fine-tuning")
    if backend == "openai":
        s.require("openai_api_key")
        result = hosted.openai_finetune(s.openai_api_key, s.hosted_base_model, root / "train.jsonl",
                                        root / "eval.jsonl", s.train_epochs)
    elif backend == "together":
        s.require("together_api_key")
        result = hosted.together_finetune(s.together_api_key, s.hosted_base_model, root / "train.jsonl",
                                          root / "eval.jsonl", s.train_epochs, s.lora_rank)
    else:
        raise typer.Exit(f"Unknown backend {backend}")
    typer.echo("Done. To serve it, set in .env:\n" + "\n".join(f"{k}={v}" for k, v in result.items()))


@app.command()
def promote(run_id: str):
    """Make a trained run the one the GPU endpoint serves (takes effect on next container start)."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(run_id)
    _modal_volume("put", "--force", VOLUME, f.name, "/models/CURRENT")
    if get_settings().serve_gpu_snapshot:
        typer.echo(f"Marked {run_id} as current. Run `uv run ajaanai deploy` to serve it — the GPU snapshot "
                   "still holds the previous model until a redeploy takes a new one.")
    else:
        typer.echo(f"Serving {run_id} from the next cold start. Force it now with: "
                   f"uv run modal app stop {APP_NAME} && uv run ajaanai deploy")


@app.command("eval")
def eval_cmd(
    arm: list[str] = typer.Option(..., help="name=https://.../v1 or name=https://.../v1|model (repeatable)"),
    n: int = typer.Option(100, help="Questions to sample"),
    questions: Path = typer.Option(None, help="Defaults to data/dataset/eval_questions.jsonl"),
):
    """Judge arms (endpoints) on held-out questions for style and fidelity."""
    _setup_logging()
    from .train.eval import Arm, run_eval

    s = get_settings()
    s.require("anthropic_api_key")
    qpath = questions or s.data_dir / "dataset" / "eval_questions.jsonl"
    arms = [Arm.parse(a, s.endpoint_bearer_secret) for a in arm]
    summary = run_eval(qpath, arms, s.judge_model, s.anthropic_api_key, n=n, out_dir=s.data_dir / "eval")
    for name, r in summary.items():
        typer.echo(f"{name:20} style={r['style']} fidelity={r['fidelity']} n={r['n']} errors={r['errors']}")


# --- talking to it -------------------------------------------------------------------------------


@app.command()
def chat(
    url: Optional[str] = typer.Option(None, help="Proxy base URL (default ENDPOINT_URL)"),
):
    """Text conversation with the deployed endpoint, streamed (fillers shown dimmed)."""
    import httpx

    s = get_settings()
    base = (url or s.endpoint_url or "").rstrip("/")
    if not base:
        raise typer.Exit("Set ENDPOINT_URL (see `ajaanai endpoints`) or pass --url")
    headers = {"Authorization": f"Bearer {s.endpoint_bearer_secret}"} if s.endpoint_bearer_secret else {}
    from .dataset.persona import FILLERS_FIRST, FILLERS_REPEAT

    fillers = set(s.fillers_first or FILLERS_FIRST) | set(s.fillers_repeat or FILLERS_REPEAT)
    history: list[dict] = []
    typer.echo("Ctrl-D to quit. (First reply may take a minute while the GPU wakes.)")
    while True:
        try:
            q = input("\n> ")
        except EOFError:
            return
        history.append({"role": "user", "content": q})
        answer = []
        with httpx.stream("POST", f"{base}/v1/chat/completions", headers=headers, timeout=300,
                          json={"model": "ajaan", "stream": True, "messages": history}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                piece = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                if piece in fillers:
                    typer.secho(piece, dim=True, nl=False)
                else:
                    answer.append(piece)
                    typer.secho(piece, fg="yellow", nl=False)
        typer.echo()
        history.append({"role": "assistant", "content": "".join(answer)})


# --- voice / telephony ----------------------------------------------------------------------------


@app.command("voice-clips")
def voice_clips(hours: float = 2.0, since_year: int = 2023, out: Path = Path("data/voice/pvc"),
                pull: bool = True):
    """Prepare cleaned audio for a Professional Voice Clone (to be created on the speaker's account)."""
    from .voice.elevenlabs import consent_granted

    ok, why = consent_granted(get_settings().consent_file)
    if not ok:
        typer.secho(f"Note: consent not yet recorded ({why}). Preparing clips is fine; cloning is not.",
                    fg="yellow")
    if pull:
        _pull_catalog()
    from .catalog import open_catalog
    from .voice.clips import prepare_pvc_clips

    files = prepare_pvc_clips(open_catalog(), get_settings(), out, hours=hours, since_year=since_year)
    typer.echo(f"Wrote {len(files)} files to {out}")


@app.command("agent-sync")
def agent_sync(endpoint_url: Optional[str] = typer.Option(None, help="Default ENDPOINT_URL")):
    """Create/update the ElevenLabs agent (custom LLM -> our endpoint, voice, greeting, turn-taking)."""
    from .voice.elevenlabs import sync_agent

    s = get_settings()
    url = endpoint_url or s.endpoint_url
    if not url:
        raise typer.Exit("Set ENDPOINT_URL (see `ajaanai endpoints`)")
    try:
        init = _fn("elevenlabs_init").get_web_url()
    except Exception:
        init = None
    agent_id = sync_agent(s, url, init, s.data_dir / "telephony.json")
    typer.echo(f"Agent {agent_id} synced (voice {s.agent_voice_id}{', clone' if s.agent_voice_is_clone else ''}).")


@app.command()
def phone(
    number: Optional[str] = typer.Option(None, help="Existing Twilio number (E.164). Default TWILIO_PHONE_NUMBER"),
    search: bool = typer.Option(False, help="List available Twilio numbers"),
    buy: Optional[str] = typer.Option(None, help="Buy this number from Twilio (charges your account)"),
    area_code: Optional[str] = typer.Option(None),
):
    """Connect a Twilio number to the agent (optionally search for / buy one)."""
    from .voice import elevenlabs as el

    s = get_settings()
    if search:
        for n in el.search_twilio_numbers(s, area_code=area_code):
            typer.echo(n)
        return
    if buy:
        typer.confirm(f"Buy {buy} on your Twilio account (about $1.15/month)?", abort=True)
        number = el.buy_twilio_number(s, buy)
        typer.echo(f"Bought {number}. Add TWILIO_PHONE_NUMBER={number} to .env")
    number = number or s.twilio_phone_number
    if not number:
        raise typer.Exit("Pass --number, or --search / --buy")
    pid = el.connect_number(s, number, s.data_dir / "telephony.json")
    typer.echo(f"{number} now answers with the agent (ElevenLabs phone id {pid}).")


if __name__ == "__main__":
    app()

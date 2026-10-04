# ajaanAI

Talk by phone with an AI trained on the Dhamma talks and writings of Thanissaro Bhikkhu
(Ajaan Geoff). The talks and writings are freely available at [dhammatalks.org](https://www.dhammatalks.org).

```
dhammatalks.org ──► catalog ──► transcripts ──► training data ──► fine-tuned model ──┐
  (evening talks daily,  (SQLite)  (site, else ASR)  (Claude writes the    (any open or     │
   other sections on                                 questions; answers    hosted LLM)      │
   demand)                                           are his own words)                    ▼
                                                                         RAG + filler proxy (Modal)
                                                                                    ▲
                                    caller ──► Twilio number ──► ElevenLabs Agent ──┘  (voice)
```

> **Not Ajaan Geoff.** The agent says up front that it is an AI. The source texts are licensed
> [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/), and the author treats *any* sale
> as commercial use, so keep this free. A cloned voice is used only with documented permission
> (see [CONSENT.md](CONSENT.md)).

## Setup

```bash
uv sync                      # Python 3.11+
cp .env.example .env         # fill in keys (see below)
modal token new              # once
ajaanai secrets-sync         # uploads .env to the Modal secret "ajaanai"
ajaanai deploy               # cron jobs, GPU workers, endpoints
```

## 1. Collect data

There are two separate components:

| | What | How it runs |
|---|---|---|
| **Evening talks** | New talks. Re-checks recent ones until the site posts an official transcript. Transcribes the rest after a grace period (21 days by default). | Daily Modal cron (`EVENING_CRON`), or `ajaanai evening` |
| **Everything else** | Books, morning talks, lectures, guided meditations, essays | `ajaanai collect [names]`, configured in [`sources.yaml`](sources.yaml) |

```bash
ajaanai evening --backfill            # one-off crawl of 2000–today (hours; runs on Modal)
ajaanai collect --list                # sources and their on/off state
ajaanai collect books                 # or: collect lectures morning_talks ...
ajaanai collect books --dry-run       # show what would be fetched
ajaanai status                        # catalog counts
```

Official transcripts always win. If ASR text exists and the site later posts a transcript, the
transcript replaces it. Text that is not his own (e.g. his translations of Ajaan Lee) is labelled
`author: translation`. It is used for retrieval only, never to train his speaking style.

**Transcription provider:** set `TRANSCRIBE_PROVIDER` plus that provider's key. Nothing else changes.

| Provider | Key | Notes |
|---|---|---|
| `elevenlabs` (default) | `ELEVENLABS_API_KEY` | Scribe v2. Reuses the phone agent's key. |
| `openai` | `OPENAI_API_KEY` | gpt-4o-transcribe |
| `deepgram` | `DEEPGRAM_API_KEY` | nova-3 with Pali key terms |
| `assemblyai` | `ASSEMBLYAI_API_KEY` | |
| `modal_whisper` | — | Self-hosted faster-whisper on a Modal GPU. Cheapest at volume. |

To compare providers on a few talks: `ajaanai transcribe --provider deepgram --limit 3`.

## 2. Build training data

```bash
ajaanai build-dataset --sample 50     # quick look: data/dataset/sample/train.jsonl
ajaanai build-dataset                 # full run via the Claude Batches API (50% cheaper, resumable)
```

Claude writes the questions a caller might ask of each passage. It also picks which of his
**own sentences** make up a short spoken answer. Every assistant turn is therefore in his words.
The output is provider-neutral chat JSONL, split by document into train and eval sets.

## 3. Fine-tune (model is swappable)

```bash
ajaanai models                        # profiles: qwen3.5-27b (default), qwen3.5-9b, qwen3-32b, llama…, generic
ajaanai train                         # TRAIN_BACKEND=modal: LoRA on Modal (H100 for 27B)
ajaanai eval --arm tuned=$ENDPOINT_URL/v1 --arm base=https://…/v1
ajaanai promote <run_id>              # the GPU endpoint serves it from the next cold start
```

How to swap the model:
- **Any Hugging Face model:** `MODEL_PROFILE=generic MODEL_ID=org/name`. Add a profile in
  [`models.py`](src/ajaanai/models.py) only if the model has special needs (thinking toggle,
  tool parser, GPU size).
- **Hosted fine-tuning:** `TRAIN_BACKEND=openai` or `together` with `HOSTED_BASE_MODEL=…`. It
  prints the `UPSTREAM_*` values to serve the result.
- **Any OpenAI-compatible API as the brain:** `LLM_BACKEND=openai_compatible` plus `UPSTREAM_*`.
  The same RAG and filler proxy runs in front of it on a cheap CPU container.

## 4. Serve and talk to it

```bash
ajaanai endpoints                     # put the LLM URL in .env as ENDPOINT_URL
ajaanai chat                          # text conversation; fillers shown dimmed
```

The proxy is an OpenAI-compatible `/v1/chat/completions`. For each turn it:
- retrieves the most relevant passages from his talks (`RAG_TOP_K`);
- decides whether to think (`REASONING_MODE`: `off` | `on` | `auto`);
- removes any reasoning from what the caller hears;
- speaks short fillers in his register ("Hmm…", "Let's look at that…") every `FILLER_INTERVAL_S`
  seconds while the model thinks (`FILLER_MODE`).

The GPU scales to zero. To keep cold starts short:
- **GPU snapshot (on by default, `SERVE_GPU_SNAPSHOT`).** Each deploy boots vLLM once, warms it
  up, puts it to sleep, and Modal snapshots the container. Cold starts restore that snapshot
  instead of booting, which should take roughly 10–30 s rather than 1–2 min. The first cold start
  after a deploy is still slow while the snapshot is taken. The model is frozen into the snapshot,
  so **redeploy after `ajaanai promote`** or after changing serving settings. The retrieval index
  is loaded fresh on every start.
- **Wake-up on call.** ElevenLabs' conversation-start webhook wakes the GPU as the call connects,
  and the greeting covers part of the wait.
- **Always warm.** For instant answers, set `SERVE_MIN_CONTAINERS=1` (about $2/hour on an L40S).

## 5. Phone

```bash
ajaanai agent-sync                    # ElevenLabs agent: custom LLM -> our endpoint, voice, greeting
ajaanai phone --search --area-code 415
ajaanai phone --buy +14155550123      # or --number <a number you already own>
```

The caller can pause to think: `AGENT_TURN_TIMEOUT_S` defaults to 7 s.

### Voice (gated on consent)

The agent uses a stock voice until permission is recorded in [CONSENT.md](CONSENT.md).
- `ajaanai voice-clips --hours 2` prepares cleaned recordings for a Professional Voice Clone.
- The clone itself has to be created and verified on the speaker's own ElevenLabs account, then
  shared with yours.
- After that, set `AGENT_VOICE_ID=<clone id>` and `AGENT_VOICE_IS_CLONE=true`, and run
  `ajaanai agent-sync`.

## Costs (rough)

| Item | Cost |
|---|---|
| Transcription backlog (~700 h) | $100–300 with a hosted API, ~$15 with `modal_whisper` |
| Dataset synthesis | ~$20–60 (Haiku via the Batches API) |
| 27B LoRA run | ~$15–40 |
| Per call minute | ElevenLabs ~$0.08–0.10 + Twilio ~$0.01 + GPU ~$0.03 while active |
| Idle | $0 + $1.15/month for the number |

## Development

```bash
uv sync --group dev && uv run pytest
ajaanai evening --local --limit 5 --no-transcribe   # scrape into ./data without Modal
```

Layout: `scrape/` (two components + parsers), `transcribe/` (providers), `dataset/` (persona,
chunking, synthesis), `rag/`, `train/` (LoRA, hosted, eval), `serve/proxy.py`, `voice/`
(ElevenLabs/Twilio, clip prep), `models.py` (profiles), `modal_app.py` (all cloud functions).

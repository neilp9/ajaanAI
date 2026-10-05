"""ElevenLabs Agent + Twilio number setup, and the voice-clone consent gate.

State (agent id, secret id, phone number id) is kept in data/telephony.json so re-running
`ajaanai agent-sync` updates the same agent instead of creating new ones.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from ..config import Settings
from ..dataset.persona import GREETING, SYSTEM_PROMPT

EL = "https://api.elevenlabs.io/v1"
TWILIO = "https://api.twilio.com/2010-04-01"


# --- consent gate ----------------------------------------------------------------------------


def consent_granted(path: Path) -> tuple[bool, str]:
    """CONSENT.md must say `status: granted` and point at the written permission."""
    if not path.exists():
        return False, f"{path} not found"
    text = path.read_text()
    status = re.search(r"^status:\s*(\S+)", text, re.MULTILINE | re.IGNORECASE)
    doc = re.search(r"^documentation:\s*(\S.*)$", text, re.MULTILINE | re.IGNORECASE)
    if not status or status.group(1).lower() != "granted":
        return False, "CONSENT.md status is not 'granted'"
    if not doc or doc.group(1).strip().lower() in {"", "-", "none", "todo"}:
        return False, "CONSENT.md has no documentation reference"
    return True, "ok"


def require_voice_consent(settings: Settings) -> None:
    if not settings.agent_voice_is_clone:
        return
    ok, why = consent_granted(settings.consent_file)
    if not ok:
        raise SystemExit(
            f"Refusing to use a cloned voice: {why}.\n"
            "A clone of Thanissaro Bhikkhu's voice needs his (or Metta Forest Monastery's) documented "
            "permission, and ElevenLabs requires a Professional clone to be created and verified on the "
            "speaker's own account, then shared with yours. Record that in CONSENT.md first.")


# --- state -----------------------------------------------------------------------------------


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def __getitem__(self, k):
        return self.data.get(k)

    def __setitem__(self, k, v):
        self.data[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def _ok(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        raise SystemExit(f"{r.request.method} {r.request.url} -> HTTP {r.status_code}: {r.text[:800]}")
    return r.json() if r.content else {}


# --- ElevenLabs agent -------------------------------------------------------------------------


def agent_body(settings: Settings, llm_url: str, secret_id: str, init_webhook_url: str | None) -> dict:
    body = {
        "name": settings.agent_name,
        "conversation_config": {
            "agent": {
                "prompt": {
                    "prompt": SYSTEM_PROMPT,
                    "llm": "custom-llm",
                    "custom_llm": {"url": llm_url, "model_id": "ajaan-voice", "api_key": {"secret_id": secret_id}},
                },
                "first_message": GREETING,
                "language": "en",
            },
            "tts": {"voice_id": settings.agent_voice_id, "model_id": settings.agent_tts_model},
            "turn": {
                "turn_timeout": settings.agent_turn_timeout_s,
                "silence_end_call_timeout": settings.agent_silence_end_call_s,
                "turn_eagerness": "patient",
            },
            "conversation": {"max_duration_seconds": settings.agent_max_call_s},
        },
    }
    if init_webhook_url:
        body["platform_settings"] = {
            "workspace_overrides": {"conversation_initiation_client_data_webhook": {
                "url": init_webhook_url, "request_headers": {}}},
            "overrides": {"enable_conversation_initiation_client_data_from_webhook": True},
        }
    return body


def sync_agent(settings: Settings, endpoint_url: str, init_webhook_url: str | None, state_path: Path) -> str:
    settings.require("elevenlabs_api_key", "endpoint_bearer_secret")
    require_voice_consent(settings)
    h = {"xi-api-key": settings.elevenlabs_api_key}
    st = State(state_path)
    if not st["secret_id"]:
        st["secret_id"] = _ok(httpx.post(f"{EL}/convai/secrets", headers=h, timeout=30, json={
            "type": "new", "name": "ajaanai-endpoint", "value": settings.endpoint_bearer_secret}))["secret_id"]
    body = agent_body(settings, endpoint_url.rstrip("/") + "/v1", st["secret_id"], init_webhook_url)
    if st["agent_id"]:
        _ok(httpx.patch(f"{EL}/convai/agents/{st['agent_id']}", headers=h, json=body, timeout=30))
    else:
        st["agent_id"] = _ok(httpx.post(f"{EL}/convai/agents/create", headers=h, json=body, timeout=30))["agent_id"]
    return st["agent_id"]


# --- Twilio number ------------------------------------------------------------------------------


def search_twilio_numbers(settings: Settings, country: str = "US", area_code: str | None = None) -> list[str]:
    settings.require("twilio_account_sid", "twilio_auth_token")
    auth = (settings.twilio_account_sid, settings.twilio_auth_token)
    params = {"VoiceEnabled": "true", "PageSize": 5} | ({"AreaCode": area_code} if area_code else {})
    r = _ok(httpx.get(f"{TWILIO}/Accounts/{settings.twilio_account_sid}/AvailablePhoneNumbers/{country}/Local.json",
                      auth=auth, params=params, timeout=30))
    return [n["phone_number"] for n in r.get("available_phone_numbers", [])]


def buy_twilio_number(settings: Settings, number: str) -> str:
    auth = (settings.twilio_account_sid, settings.twilio_auth_token)
    r = _ok(httpx.post(f"{TWILIO}/Accounts/{settings.twilio_account_sid}/IncomingPhoneNumbers.json",
                       auth=auth, data={"PhoneNumber": number, "FriendlyName": "ajaanAI"}, timeout=30))
    return r["phone_number"]


def connect_number(settings: Settings, number: str, state_path: Path) -> str:
    """Import a Twilio number into ElevenLabs and route it to the agent."""
    settings.require("elevenlabs_api_key", "twilio_account_sid", "twilio_auth_token")
    st = State(state_path)
    if not st["agent_id"]:
        raise SystemExit("Run `ajaanai agent-sync` first")
    h = {"xi-api-key": settings.elevenlabs_api_key}
    if st["phone_number"] != number or not st["phone_number_id"]:
        st["phone_number_id"] = _ok(httpx.post(f"{EL}/convai/phone-numbers", headers=h, timeout=30, json={
            "phone_number": number, "label": "ajaanAI", "provider": "twilio",
            "sid": settings.twilio_account_sid, "token": settings.twilio_auth_token,
            "agent_id": st["agent_id"]}))["phone_number_id"]
        st["phone_number"] = number
    _ok(httpx.patch(f"{EL}/convai/phone-numbers/{st['phone_number_id']}", headers=h, timeout=30,
                    json={"agent_id": st["agent_id"]}))
    return st["phone_number_id"]

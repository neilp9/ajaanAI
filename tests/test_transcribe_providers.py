"""Request-shape tests for each hosted ASR provider (HTTP mocked with respx)."""

import json

import httpx

from ajaanai.transcribe.base import GLOSSARY, absolute_audio_url, paragraphize
from ajaanai.transcribe.providers import AssemblyAI, Deepgram, ElevenLabsScribe

URL = "https://www.dhammatalks.org/Archive/y2005/051231%20Proving%20the%20Teachings.mp3"


def test_elevenlabs_multipart_with_repeated_keyterms(respx_mock):
    route = respx_mock.post("https://api.elevenlabs.io/v1/speech-to-text").mock(
        return_value=httpx.Response(200, json={"text": "The breath is home."}))
    assert ElevenLabsScribe("k").transcribe(URL, GLOSSARY) == "The breath is home."
    req = route.calls.last.request
    body = req.read().decode()
    assert req.headers["content-type"].startswith("multipart/form-data")
    assert req.headers["xi-api-key"] == "k"
    assert 'name="model_id"\r\n\r\nscribe_v2' in body
    assert f'name="cloud_storage_url"\r\n\r\n{URL}' in body
    assert body.count('name="keyterms"') == len([t for t in GLOSSARY if len(t) <= 50])
    assert "jhāna" in body
    assert "filename=" not in body  # plain form fields, not file uploads


def test_deepgram_url_and_keyterms(respx_mock):
    route = respx_mock.post("https://api.deepgram.com/v1/listen").mock(return_value=httpx.Response(200, json={
        "results": {"channels": [{"alternatives": [{"transcript": "flat", "paragraphs": {"paragraphs": [
            {"sentences": [{"text": "One."}, {"text": "Two."}]}, {"sentences": [{"text": "Three."}]}]}}]}]}}))
    assert Deepgram("k").transcribe(URL, ["jhāna", "Dhamma"]) == "One. Two.\n\nThree."
    req = route.calls.last.request
    assert req.headers["authorization"] == "Token k"
    assert json.loads(req.content) == {"url": URL}
    assert req.url.params.get_list("keyterm") == ["jhāna", "Dhamma"]
    assert req.url.params["model"] == "nova-3"


def test_assemblyai_submit_poll_paragraphs(respx_mock):
    base = "https://api.assemblyai.com/v2"
    submit = respx_mock.post(f"{base}/transcript").mock(return_value=httpx.Response(200, json={"id": "t1"}))
    respx_mock.get(f"{base}/transcript/t1").mock(side_effect=[
        httpx.Response(200, json={"status": "processing"}),
        httpx.Response(200, json={"status": "completed", "text": "flat"})])
    respx_mock.get(f"{base}/transcript/t1/paragraphs").mock(
        return_value=httpx.Response(200, json={"paragraphs": [{"text": "A."}, {"text": "B."}]}))
    assert AssemblyAI("k", poll_s=0).transcribe(URL, ["jhāna"]) == "A.\n\nB."
    sent = json.loads(submit.calls.last.request.content)
    assert sent["audio_url"] == URL and sent["word_boost"] == ["jhāna"]


def test_audio_url_quoting_and_paragraphize():
    assert absolute_audio_url("https://www.dhammatalks.org", "/Archive/y2005/051231 Proving the Teachings.mp3") == URL
    text = paragraphize(" ".join(["This is one sentence."] * 60), target_words=40)
    assert text.count("\n\n") >= 3

import json
from types import SimpleNamespace as NS

from ajaanai.catalog import Doc
from ajaanai.dataset.chunk import passages_for
from ajaanai.dataset.synth import SynthStore, render, synth_batch, synth_sync

OUT = {"usable": True, "topics": ["breath"], "items": [{"question": "How do I begin?", "short_answer_sentences": [0, 1]}]}


def message(payload, stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="text", text=json.dumps(payload))])


def passages():
    doc = Doc(id="d", source="evening", kind="talk", title="T", page_url="/x",
              text="\n\n".join(" ".join(f"Sentence {p}-{i} goes here." for i in range(30)) for p in range(16)))
    return passages_for(doc)


def test_render_numbers_sentences():
    out = render(passages()[0])
    assert out.startswith("Title: T") and "[0] Sentence 0-0 goes here." in out


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        assert kw["output_config"]["format"]["type"] == "json_schema"
        return message(OUT)


def test_synth_sync_and_resume(tmp_path):
    client = NS(messages=FakeMessages())
    store = SynthStore(tmp_path)
    ps = passages()
    assert synth_sync(client, "m", ps, store) == len(ps)
    assert store.done_ids() == {p.id for p in ps}


def test_synth_batch_resumable(tmp_path):
    ps = passages()
    created = []

    class Batches:
        def create(self, requests):
            created.append(requests)
            return NS(id="b1")

        def retrieve(self, bid):
            return NS(processing_status="ended", request_counts=NS(processing=0))

        def results(self, bid):
            reqs = created[0]
            yield NS(custom_id=reqs[0]["custom_id"], result=NS(type="succeeded", message=message(OUT)))
            yield NS(custom_id=reqs[1]["custom_id"], result=NS(type="errored", message=None))
            yield NS(custom_id=reqs[2]["custom_id"], result=NS(type="succeeded", message=message(OUT, "max_tokens")))

    client = NS(messages=NS(batches=Batches()))
    store = SynthStore(tmp_path)
    assert synth_batch(client, "m", ps, store, poll_s=0) == 1
    assert store.pending_batches() == []
    assert len(store.done_ids()) == 1
    # a second run only submits the passages still missing
    created.clear()
    synth_batch(client, "m", ps, store, poll_s=0)
    assert len(created[0]) == len(ps) - 1

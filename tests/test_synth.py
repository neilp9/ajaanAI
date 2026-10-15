import json
from types import SimpleNamespace as NS

from ajaanai.catalog import Doc
from ajaanai.dataset.chunk import passages_for
from ajaanai.dataset.synth import SynthStore, _params, input_sha, render, run_sync, synth_batch, synth_sync

OUT = {"usable": True, "topics": ["breath"], "conversations": [{"exchanges": [
    {"caller": "How do I begin?", "sentences": [0, 1], "longer": False, "ask_back": ""}]}]}


def message(payload, stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="text", text=json.dumps(payload))])


def passages():
    doc = Doc(id="d", source="evening", kind="talk", title="T", page_url="/x",
              text="\n\n".join(" ".join(f"Sentence {p}-{i} goes here." for i in range(30)) for p in range(16)))
    return passages_for(doc)


def test_render_numbers_sentences():
    out = render(passages()[0])
    assert out.startswith("Channel: ") and "Title: T" in out and "[0] Sentence 0-0 goes here." in out


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
    assert store.todo([(p.id, _params("m", p)) for p in ps]) == []
    assert set(store.load()) == {p.id for p in ps}


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
    assert len(store.load()) == 1
    # a second run only submits the passages still missing
    created.clear()
    synth_batch(client, "m", ps, store, poll_s=0)
    assert len(created[0]) == len(ps) - 1


def _items(ps):
    return [(p.id, _params("m", p)) for p in ps]


def test_changed_input_is_redone(tmp_path):
    """A passage whose text changed (e.g. ASR replaced by the site transcript) keeps its id but is
    synthesized again; its old result, numbered against the old sentences, is never used."""
    client = NS(messages=FakeMessages())
    store = SynthStore(tmp_path)
    ps = passages()
    run_sync(client, _items(ps), store)
    client.messages.calls.clear()

    ps[1].text = ps[1].text.replace("Sentence", "Clause")
    items = _items(ps)
    assert [i for i, _ in store.todo(items)] == [ps[1].id]
    assert run_sync(client, items, store) == 1
    assert len(client.messages.calls) == 1
    expected = {i: input_sha(params) for i, params in items}
    assert store.load(expected)[ps[1].id]["input_sha"] == expected[ps[1].id]
    assert store.todo(items) == []


def test_failed_redo_never_falls_back_to_stale_result(tmp_path):
    client = NS(messages=FakeMessages())
    store = SynthStore(tmp_path)
    ps = passages()
    run_sync(client, _items(ps), store)

    ps[0].text = ps[0].text.replace("Sentence", "Clause")
    items = _items(ps)
    client.messages.create = lambda **kw: message(OUT, "max_tokens")  # unparseable
    assert run_sync(client, items, store) == 0
    loaded = store.load({i: input_sha(params) for i, params in items})
    assert ps[0].id not in loaded and ps[1].id in loaded
    assert [i for i, _ in store.todo(items)] == [ps[0].id]  # retried next run


def test_rows_saved_before_hashing_count_as_current(tmp_path):
    store = SynthStore(tmp_path)
    ps = passages()
    with store.results.open("w") as f:  # legacy rows: no input_sha
        for p in ps:
            f.write(json.dumps({"passage_id": p.id, **OUT}) + "\n")
    items = _items(ps)
    assert store.todo(items) == []
    assert set(store.load({i: input_sha(params) for i, params in items})) == {p.id for p in ps}


def test_shaped_mode_has_its_own_prompt_schema_and_directory(tmp_path):
    from ajaanai.config import Settings
    from ajaanai.dataset import dataset_dir
    from ajaanai.dataset.synth import SCHEMA, SCHEMA_SHAPED, SYSTEM, SYSTEM_SHAPED

    p = passages()[0]
    verbatim, shaped = _params("m", p), _params("m", p, shaped=True)
    assert verbatim["system"] == SYSTEM and verbatim["output_config"]["format"]["schema"] == SCHEMA
    assert shaped["system"] == SYSTEM_SHAPED and shaped["output_config"]["format"]["schema"] == SCHEMA_SHAPED
    # same messages, so the input hash can't tell the modes apart: they must not share a store
    assert input_sha(verbatim) == input_sha(shaped)
    assert dataset_dir(Settings(data_dir=tmp_path)).name == "dataset"
    assert dataset_dir(Settings(data_dir=tmp_path, dataset_reply_mode="shaped")).name == "dataset-shaped"

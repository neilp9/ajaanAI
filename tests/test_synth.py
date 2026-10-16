import hashlib
import json

from ajaanai.catalog import Doc
from ajaanai.dataset.chunk import passages_for
from ajaanai.dataset.synth import SynthStore, _params, input_sha, render, run_sync, synth_batch, synth_sync
from ajaanai.llm import BatchState, Reply

OUT = {"usable": True, "topics": ["breath"], "conversations": [{"exchanges": [
    {"caller": "How do I begin?", "sentences": [0, 1], "longer": False, "ask_back": ""}]}]}


class FakeBackend:
    """Answers every request with OUT (or `reply`); a batch is one call per item, ended at once."""
    max_per_batch = 50_000

    def __init__(self, reply=None):
        self.calls, self.batches = [], {}
        self.reply = reply or (lambda req: Reply(OUT, 10, 5))

    def complete(self, req):
        self.calls.append(req)
        return self.reply(req)

    def submit(self, items):
        bid = f"b{len(self.batches)}"
        self.batches[bid] = items
        return bid

    def status(self, bid):
        n = len(self.batches[bid])
        return BatchState(True, n, n, n, 0)

    def results(self, bid):
        for cid, req in self.batches[bid]:
            yield cid, self.reply(req)


def passages():
    doc = Doc(id="d", source="evening", kind="talk", title="T", page_url="/x",
              text="\n\n".join(" ".join(f"Sentence {p}-{i} goes here." for i in range(30)) for p in range(16)))
    return passages_for(doc)


def test_render_numbers_sentences():
    out = render(passages()[0])
    assert out.startswith("Channel: ") and "Title: T" in out and "[0] Sentence 0-0 goes here." in out


def test_synth_sync_and_resume(tmp_path):
    backend = FakeBackend()
    store = SynthStore(tmp_path)
    ps = passages()
    assert synth_sync(backend, "m", ps, store) == len(ps)
    assert backend.calls[0].model == "m" and backend.calls[0].schema["required"] == ["usable", "topics", "conversations"]
    assert store.todo([(p.id, _params("m", p)) for p in ps]) == []
    assert set(store.load()) == {p.id for p in ps}


def test_synth_batch_resumable(tmp_path):
    ps = passages()
    # first item succeeds, second fails outright, third is cut off (no data)
    outcomes = iter([Reply(OUT), None, Reply(None)] + [None] * len(ps))
    backend = FakeBackend(reply=lambda req: next(outcomes))
    store = SynthStore(tmp_path)
    assert synth_batch(backend, "m", ps, store, poll_s=0) == 1
    assert store.pending_batches() == []
    assert len(store.load()) == 1
    # a second run only submits the passages still missing
    backend.batches.clear()
    backend.reply = lambda req: Reply(OUT)
    assert synth_batch(backend, "m", ps, store, poll_s=0) == len(ps) - 1
    assert len(backend.batches["b0"]) == len(ps) - 1


def test_batches_split_at_the_backends_limit(tmp_path):
    backend = FakeBackend()
    backend.max_per_batch = 2
    ps = passages()[:5]
    assert synth_batch(backend, "m", ps, store=SynthStore(tmp_path), poll_s=0) == 5
    assert [len(b) for b in backend.batches.values()] == [2, 2, 1]
    assert [cid for b in backend.batches.values() for cid, _ in b] == ["p0", "p1", "p2", "p3", "p4"]


def _items(ps):
    return [(p.id, _params("m", p)) for p in ps]


def test_changed_input_is_redone(tmp_path):
    """A passage whose text changed (e.g. ASR replaced by the site transcript) keeps its id but is
    synthesized again; its old result, numbered against the old sentences, is never used."""
    backend = FakeBackend()
    store = SynthStore(tmp_path)
    ps = passages()
    run_sync(backend, _items(ps), store)
    backend.calls.clear()

    ps[1].text = ps[1].text.replace("Sentence", "Clause")
    items = _items(ps)
    assert [i for i, _ in store.todo(items)] == [ps[1].id]
    assert run_sync(backend, items, store) == 1
    assert len(backend.calls) == 1
    expected = {i: input_sha(params) for i, params in items}
    assert store.load(expected)[ps[1].id]["input_sha"] == expected[ps[1].id]
    assert store.todo(items) == []


def test_failed_redo_never_falls_back_to_stale_result(tmp_path):
    backend = FakeBackend()
    store = SynthStore(tmp_path)
    ps = passages()
    run_sync(backend, _items(ps), store)

    ps[0].text = ps[0].text.replace("Sentence", "Clause")
    items = _items(ps)
    backend.reply = lambda req: Reply(None)  # cut off / unparseable
    assert run_sync(backend, items, store) == 0
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
    from ajaanai.dataset import synth_dir
    from ajaanai.dataset.synth import SCHEMA, SCHEMA_SHAPED, SYSTEM, SYSTEM_SHAPED

    p = passages()[0]
    verbatim, shaped = _params("m", p), _params("m", p, shaped=True)
    assert verbatim.system == SYSTEM and verbatim.schema == SCHEMA
    assert shaped.system == SYSTEM_SHAPED and shaped.schema == SCHEMA_SHAPED
    # same messages, so the input hash can't tell the modes apart: they must not share a store
    assert input_sha(verbatim) == input_sha(shaped)
    assert synth_dir(Settings(_env_file=None, data_dir=tmp_path)).name == "verbatim_anthropic-claude-haiku-4-5"
    assert synth_dir(Settings(_env_file=None, data_dir=tmp_path, dataset_reply_mode="shaped")).name == \
        "shaped_anthropic-claude-haiku-4-5"


def test_each_build_gets_a_label_and_the_latest_is_the_default(tmp_path):
    from datetime import date

    import pytest

    from ajaanai.config import Settings
    from ajaanai.dataset import dataset_dir, new_label, synth_dir

    s = Settings(_env_file=None, data_dir=tmp_path, synth_model="openai:org/gpt-x")
    assert synth_dir(s) == tmp_path / "datasets" / "verbatim_openai-org-gpt-x"
    with pytest.raises(SystemExit, match="No dataset built"):
        dataset_dir(s)
    for day in (date(2026, 10, 9), date(2026, 10, 14), date(2026, 10, 14)):
        d = synth_dir(s) / new_label(s, day)
        d.mkdir(parents=True)
        (d / "train.jsonl").write_text("")
    builds = sorted(p.name for p in synth_dir(s).iterdir())
    assert builds == ["verbatim_openai-org-gpt-x_20261009", "verbatim_openai-org-gpt-x_20261014",
                      "verbatim_openai-org-gpt-x_20261014-2"]
    (synth_dir(s) / "sample").mkdir()  # samples are never picked as a build
    assert dataset_dir(s).name == "verbatim_openai-org-gpt-x_20261014-2"
    assert dataset_dir(s, "verbatim_openai-org-gpt-x_20261009") == synth_dir(s) / "verbatim_openai-org-gpt-x_20261009"
    with pytest.raises(SystemExit, match="No dataset"):
        dataset_dir(s, "nope")


def test_input_hash_is_unchanged_from_the_chat_message_form(tmp_path):
    """Results stored before the provider-neutral requests must still count as current."""
    p = passages()[0]
    old = [{"role": "user", "content": render(p)}]
    assert input_sha(_params("m", p)) == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()[:16]

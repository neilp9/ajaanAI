import pytest

from ajaanai.lock import CatalogBusy, CatalogLock


class FakeDict(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def test_second_writer_is_refused_until_release():
    store, t = FakeDict(), [1000.0]
    a = CatalogLock(store, "backfill", clock=lambda: t[0])
    b = CatalogLock(store, "daily", clock=lambda: t[0])
    with a.held():
        with pytest.raises(CatalogBusy, match="backfill"):
            b.acquire()
    with b.held():  # free again after release
        pass
    assert "catalog" not in store


def test_stale_lock_taken_over_but_heartbeat_keeps_it_alive():
    store, t = FakeDict(), [0.0]
    a = CatalogLock(store, "crashed", stale_after_s=100, clock=lambda: t[0])
    b = CatalogLock(store, "next", stale_after_s=100, clock=lambda: t[0])
    a.acquire()
    t[0] = 90
    a.heartbeat()
    t[0] = 150  # 60s since heartbeat: still alive
    with pytest.raises(CatalogBusy):
        b.acquire()
    t[0] = 400  # holder stopped heartbeating -> stale
    b.acquire()
    assert store["catalog"]["owner"] == "next"
    a.release()  # a late release from the old holder must not drop b's lock
    assert store["catalog"]["owner"] == "next"

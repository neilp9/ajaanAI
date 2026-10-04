"""A single-writer lock for the catalog, so scheduled and manual jobs never overlap.

Modal volumes are last-commit-wins per file: two jobs writing catalog.sqlite at once would each
work on their own copy and one would silently overwrite the other. The lock lives in a Modal
Dict (atomic put-if-absent). A holder refreshes its heartbeat at each checkpoint; a lock whose
heartbeat is older than `stale_after_s` is assumed to belong to a crashed job and is taken over.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Protocol


class KV(Protocol):
    def put(self, key, value, *, skip_if_exists: bool = False) -> bool: ...
    def get(self, key, default=None): ...
    def pop(self, key): ...


class CatalogBusy(RuntimeError):
    pass


class CatalogLock:
    KEY = "catalog"

    def __init__(self, store: KV, owner: str, stale_after_s: float = 1800,
                 clock: Callable[[], float] = time.time):
        self.store, self.owner, self.stale_after_s, self.clock = store, owner, stale_after_s, clock

    def acquire(self) -> None:
        now = self.clock()
        holder = self.store.get(self.KEY)
        if holder and now - holder["heartbeat"] > self.stale_after_s:
            self.store.pop(self.KEY)  # previous holder crashed without releasing
        if not self.store.put(self.KEY, self._value(now, now), skip_if_exists=True):
            h = self.store.get(self.KEY) or {}
            age = int(now - h.get("since", now))
            raise CatalogBusy(f"catalog is in use by {h.get('owner', '?')} (running {age // 60} min)")
        self.since = now

    def heartbeat(self) -> None:
        self.store.put(self.KEY, self._value(self.since, self.clock()))

    def release(self) -> None:
        holder = self.store.get(self.KEY)
        if holder and holder.get("owner") == self.owner:
            self.store.pop(self.KEY)

    def _value(self, since: float, beat: float) -> dict:
        return {"owner": self.owner, "since": since, "heartbeat": beat}

    @contextmanager
    def held(self):
        self.acquire()
        try:
            yield self
        finally:
            self.release()

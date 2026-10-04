"""Polite HTTP client shared by every scraper: rate limit, robots.txt, retries."""

from __future__ import annotations

import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import httpx

from ..config import Settings, get_settings


class Disallowed(Exception):
    pass


class PoliteClient:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.s = settings or get_settings()
        self.client = client or httpx.Client(
            headers={"User-Agent": self.s.user_agent},
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._last = 0.0
        self._robots: urllib.robotparser.RobotFileParser | None = None

    def url(self, path_or_url: str) -> str:
        return urljoin(self.s.site_base_url + "/", path_or_url)

    def _robots_ok(self, url: str) -> bool:
        if self._robots is None:
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = self._raw_get(self.url("/robots.txt"))
                rp.parse(resp.text.splitlines() if resp.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            self._robots = rp
        return self._robots.can_fetch(self.s.user_agent, url)

    def _raw_get(self, url: str) -> httpx.Response:
        wait = self.s.request_delay_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.client.get(url)
        finally:
            self._last = time.monotonic()

    def get(self, path_or_url: str, retries: int = 3) -> httpx.Response:
        url = self.url(path_or_url)
        if urlparse(url).netloc.endswith("dhammatalks.org") and not self._robots_ok(url):
            raise Disallowed(url)
        for attempt in range(retries + 1):
            try:
                resp = self._raw_get(url)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if attempt == retries or (status is not None and status < 500 and status != 429):
                    raise
                time.sleep(2 ** attempt * 2)
        raise AssertionError("unreachable")

    def text(self, path_or_url: str) -> str:
        return self.get(path_or_url).text

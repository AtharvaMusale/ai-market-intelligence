"""A tiny rate-limited HTTP client so every source respects its published request limits."""
from __future__ import annotations

import time

import requests


class RateLimitedSession:
    """Wraps requests.Session: enforces a minimum gap between requests and a timeout on each."""

    def __init__(self, min_interval_s: float, headers: dict[str, str] | None = None) -> None:
        self._session = requests.Session()
        self._session.headers.update(headers or {})
        self._min_interval_s = min_interval_s
        self._last_request = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        wait = self._min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        response = self._session.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response

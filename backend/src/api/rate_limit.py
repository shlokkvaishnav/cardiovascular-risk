"""Per-client rate limits for the public API.

The API key is public (the web app sends it from the browser), so it cannot
identify one visitor. This middleware limits each client address instead, with
a separate limit for each group of routes.

Limits are requests per 60 seconds. Set an environment variable to change one:

    RATE_LIMIT_AUTH_RPM      /auth/login, /auth/register        (default 10)
    RATE_LIMIT_EXTRACT_RPM   /extract (calls the Groq API)      (default 5)
    RATE_LIMIT_BATCH_RPM     /batch-predict                     (default 10)
    RATE_LIMIT_PREDICT_RPM   /predict                           (default 60)
    RATE_LIMIT_DEFAULT_RPM   every other route                  (default 120)
    TRUST_PROXY_HEADERS      "true" (default) reads the client address from
                             X-Real-IP or X-Forwarded-For. Set "false" if the
                             API is reachable without a reverse proxy.

The counters live in the memory of one process. With several workers or
replicas, each one counts on its own, so the real limit is higher.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

WINDOW_SECONDS = 60
MAX_TRACKED_CLIENTS = 10_000

# Health checks and metrics must always answer. CORS preflight (OPTIONS) is
# skipped in __call__ so that browsers can still send the real request.
EXEMPT_PATHS = frozenset({"/", "/health", "/metrics"})


@dataclass(frozen=True)
class Rule:
    name: str
    prefixes: Tuple[str, ...]
    limit: int


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def default_rules() -> List[Rule]:
    # Order matters: the first rule whose prefix matches wins.
    return [
        Rule(
            "auth",
            ("/auth/login", "/auth/register"),
            _int_env("RATE_LIMIT_AUTH_RPM", 10),
        ),
        Rule("extract", ("/extract",), _int_env("RATE_LIMIT_EXTRACT_RPM", 5)),
        Rule("batch", ("/batch-predict",), _int_env("RATE_LIMIT_BATCH_RPM", 10)),
        Rule("predict", ("/predict",), _int_env("RATE_LIMIT_PREDICT_RPM", 60)),
    ]


def client_address(scope: Scope, trust_proxy_headers: bool) -> str:
    """Return the address used to count requests for one client."""
    if trust_proxy_headers:
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope["headers"]
        }
        real_ip = headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip
        # The proxy appends the real client address at the end of the list.
        # The first entries can be set by the client, so do not use them.
        forwarded = [
            p.strip()
            for p in headers.get("x-forwarded-for", "").split(",")
            if p.strip()
        ]
        if forwarded:
            return forwarded[-1]
    client = scope.get("client")
    return client[0] if client else "unknown"


class RateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        rules: Optional[List[Rule]] = None,
        default_limit: Optional[int] = None,
        trust_proxy_headers: Optional[bool] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.rules = rules if rules is not None else default_rules()
        self.default_limit = (
            default_limit
            if default_limit is not None
            else _int_env("RATE_LIMIT_DEFAULT_RPM", 120)
        )
        self.trust_proxy_headers = (
            trust_proxy_headers
            if trust_proxy_headers is not None
            else os.getenv("TRUST_PROXY_HEADERS", "true").lower() != "false"
        )
        self.clock = clock
        self._hits: Dict[Tuple[str, str], Deque[float]] = {}

    def _match(self, path: str) -> Tuple[str, int]:
        for rule in self.rules:
            if path.startswith(rule.prefixes):
                return rule.name, rule.limit
        return "default", self.default_limit

    def _retry_after(self, key: Tuple[str, str], limit: int) -> Optional[int]:
        """Count one request. Return None if allowed, or seconds to wait if not."""
        now = self.clock()
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            return max(1, math.ceil(WINDOW_SECONDS - (now - hits[0])))
        hits.append(now)
        if len(self._hits) > MAX_TRACKED_CLIENTS:
            self._drop_idle(now)
        return None

    def _drop_idle(self, now: float) -> None:
        for key in [
            k for k, h in self._hits.items() if not h or h[-1] <= now - WINDOW_SECONDS
        ]:
            del self._hits[key]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] == "OPTIONS"
            or scope["path"] in EXEMPT_PATHS
        ):
            await self.app(scope, receive, send)
            return

        group, limit = self._match(scope["path"])
        key = (client_address(scope, self.trust_proxy_headers), group)
        retry_after = self._retry_after(key, limit)
        if retry_after is not None:
            response = JSONResponse(
                {"detail": "Too many requests. Try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

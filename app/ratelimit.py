"""Per-key rate limiting.

Fairness matters here because all analysis runs through a single Stockfish
process behind a global lock — without a cap, one busy customer serializes
everyone. We key the limit on the authenticated API key (falling back to client
IP for unauthenticated routes) rather than IP alone.

This uses slowapi's in-process limiter, which is correct for a single-worker
deployment (the current HF Spaces setup). If you scale to multiple workers,
point slowapi at a shared Redis backend so the counter is global.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import RATE_LIMIT_PER_MIN


def _key_func(request: Request) -> str:
    """Rate-limit bucket: the caller's API key if present, else their IP.

    We read the raw Authorization header rather than the resolved Principal so
    the limiter can run as an ASGI-level check before route dependencies."""
    auth = request.headers.get("authorization", "")
    parts = auth.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return "key:" + parts[1].strip()
    return "ip:" + get_remote_address(request)


limiter = Limiter(key_func=_key_func, default_limits=[f"{RATE_LIMIT_PER_MIN}/minute"])

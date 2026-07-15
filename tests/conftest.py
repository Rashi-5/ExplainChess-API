"""Shared test fixtures.

Uses a throwaway SQLite DB and a session-scoped TestClient — the pipeline load
(Stockfish + 593 MB model) is slow, so it's shared across the whole test module.
Requires a Stockfish binary on PATH or STOCKFISH_PATH (Homebrew ships SF 18).
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Point the app at an isolated SQLite DB before any app import reads config.
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db.name}"
os.environ.setdefault("ADMIN_TOKEN", "test-admin")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

# A known Scholar's-mate position: white to move, Qxf7# — an "easy" puzzle.
FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
MOVES = "f3f7"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def api_key(client):
    """Mint a high-quota key once for the module."""
    r = client.post(
        "/v1/keys",
        headers={"x-admin-token": "test-admin"},
        json={"email": "test@club.com", "daily_quota": 0},
    )
    assert r.status_code == 200, r.text
    return r.json()["api_key"]


@pytest.fixture
def auth(api_key):
    return {"Authorization": f"Bearer {api_key}"}

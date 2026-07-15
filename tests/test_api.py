"""Smoke tests for the ExplainChess API surface: auth, analyze, batch, quota."""

from __future__ import annotations

from tests.conftest import FEN, MOVES

ANALYZE_KEYS = {"difficulty", "quality", "explanation", "input_interpretation", "diagnostics"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["pipeline_ready"] is True


def test_analyze_requires_key(client):
    r = client.post("/v1/analyze", json={"fen": FEN, "moves": MOVES})
    assert r.status_code == 401


def test_analyze_rejects_bad_key(client):
    r = client.post(
        "/v1/analyze",
        headers={"Authorization": "Bearer eck_not_a_real_key"},
        json={"fen": FEN, "moves": MOVES},
    )
    assert r.status_code == 401


def test_analyze_ok(client, auth):
    r = client.post("/v1/analyze", headers=auth, json={"fen": FEN, "moves": MOVES})
    assert r.status_code == 200, r.text
    body = r.json()
    assert ANALYZE_KEYS.issubset(body.keys())
    # Scholar's mate is easy at every band.
    assert body["difficulty"]["label"] == "easy"


def test_analyze_empty_moves_422(client, auth):
    r = client.post("/v1/analyze", headers=auth, json={"fen": FEN, "moves": ""})
    assert r.status_code == 422


def test_batch(client, auth):
    r = client.post(
        "/v1/analyze/batch",
        headers=auth,
        json={"puzzles": [{"fen": FEN, "moves": MOVES}, {"fen": FEN, "moves": ""}]},
    )
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == 2
    # First item succeeds, second (empty moves) errors — isolation within a batch.
    assert results[0]["result"] is not None and results[0]["error"] is None
    assert results[1]["result"] is None and results[1]["error"] is not None


def test_batch_over_cap_422(client, auth, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "MAX_BATCH", 1)
    r = client.post(
        "/v1/analyze/batch",
        headers=auth,
        json={"puzzles": [{"fen": FEN, "moves": MOVES}, {"fen": FEN, "moves": MOVES}]},
    )
    assert r.status_code == 422


def test_quota_enforced(client):
    # Fresh key with a tiny quota; the third analyze-equivalent call is rejected.
    r = client.post(
        "/v1/keys",
        headers={"x-admin-token": "test-admin"},
        json={"email": "quota@club.com", "daily_quota": 2},
    )
    key = r.json()["api_key"]
    h = {"Authorization": f"Bearer {key}"}

    assert client.post("/v1/analyze", headers=h, json={"fen": FEN, "moves": MOVES}).status_code == 200
    assert client.post("/v1/analyze", headers=h, json={"fen": FEN, "moves": MOVES}).status_code == 200
    assert client.post("/v1/analyze", headers=h, json={"fen": FEN, "moves": MOVES}).status_code == 429


def test_mint_requires_admin(client):
    r = client.post("/v1/keys", headers={"x-admin-token": "wrong"}, json={"email": "x@y.com"})
    assert r.status_code == 403

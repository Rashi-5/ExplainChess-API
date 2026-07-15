---
title: ExplainChess API
emoji: ♟️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# ExplainChess API

**Per-player chess puzzle difficulty — as a metered HTTP API.**

Send a puzzle (position + solution moves) and get back a *per-player
solve-probability curve* (how likely a player at each rating band from 1050 to
2050 is to solve it), a difficulty label, a per-band SHAP explanation of *why*
it's hard, and an engine quality/soundness check — all from the raw position, no
dataset lookup required.

Built for chess coaches and platforms that need to grade puzzle sets at the
right difficulty for their students.

---

## Authentication

Every endpoint except `/health` requires an API key as a Bearer token:

```
Authorization: Bearer eck_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Keys are issued per customer. Contact us to get one (see pricing below).

---

## Endpoints

Interactive reference (OpenAPI) is served at **`/docs`**.

### `POST /v1/analyze` — one puzzle

```bash
curl -s https://<host>/v1/analyze \
  -H "Authorization: Bearer $EC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "moves": "f3f7"
  }'
```

**`moves`** is required: space-separated UCI moves in Lichess convention —
opponent's setup move first, then the solution. If the FEN is already the
solver's position (no setup move), prefix with `0000`, e.g.
`"0000 d8h4 g2g3 h4e4"`. Empty moves are rejected (422) — the model relies on
the solution line.

**Response** (abridged):

```json
{
  "difficulty": {
    "rapid_curve": { "1050": {"p_solve": 0.89, "uncertainty_pp": 4.1}, "...": "..." },
    "p_solve_1050": 0.889,
    "p_solve_2050": 0.985,
    "skill_gap": 0.096,
    "label": "easy",
    "curve_reliability": 0.92,
    "model_mean_uncertainty_pp": 5.3
  },
  "quality":   { "verdict": "...", "confidence": "high", "margin_cp": 900, "...": "..." },
  "explanation": {
    "factors_lower_rated_1050": [ {"feature": "...", "description": "...", "shap": 0.12, "effect": "easier"} ],
    "factors_higher_rated_2050": [ "..." ],
    "skill_gap_drivers": [ "..." ]
  },
  "input_interpretation": { "mode": "lichess", "auto_corrected": false, "note": null },
  "diagnostics": { "engine": "Stockfish 18", "extract_ok": true, "...": "..." }
}
```

### `POST /v1/analyze/batch` — a whole puzzle set

Grade up to **100** puzzles in one call. A bad puzzle in the set errors only
that item — the rest still return.

```bash
curl -s https://<host>/v1/analyze/batch \
  -H "Authorization: Bearer $EC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "puzzles": [
      {"fen": "<fen1>", "moves": "<moves1>"},
      {"fen": "<fen2>", "moves": "<moves2>"}
    ]
  }'
```

```json
{
  "results": [
    {"index": 0, "result": { "difficulty": "...", "...": "..." }, "error": null},
    {"index": 1, "result": null, "error": "Solution moves are required. ..."}
  ]
}
```

### `POST /v1/solve` — derive a solution line from a bare position

For a position with no known solution, the engine plays out its principal
variation. The returned `moves` is directly usable as the `moves` field of
`/v1/analyze`.

```bash
curl -s https://<host>/v1/solve \
  -H "Authorization: Bearer $EC_API_KEY" -H "Content-Type: application/json" \
  -d '{"fen": "<fen>", "plies": 6}'
```

### Calibration (optional)

- `GET /v1/calibration/puzzles?n=15` — high-discrimination puzzles for a
  user-rating calibration session.
- `POST /v1/calibration/estimate` — given a solve history
  (`[{"curve": [...11 floats], "solved": 0|1}]`), returns an estimated user
  rating and posterior.

### `GET /health`

Unauthenticated liveness check: `{"status": "ok", "pipeline_ready": true}`.

---

## Rate limits & quotas

- **Rate limit:** requests per minute per key (default 30). Exceeding it
  returns `429`.
- **Daily quota:** optional per-customer cap on analyze-equivalent calls per day
  (each batch puzzle counts as one). Exceeding it returns `429`. A quota of `0`
  means unlimited.

---

## Errors

| Status | Meaning |
|--------|---------|
| `401`  | Missing / invalid / revoked API key |
| `422`  | Bad input (e.g. empty moves, illegal FEN, batch over cap) |
| `429`  | Rate limit or daily quota exceeded |
| `503`  | Pipeline not ready (starting up) |

---

## Pricing

Usage-based, billed per analysis (batch puzzles counted individually).
**Contact us for pilot access and pricing.**

---

## Running it yourself

```bash
pip install -r requirements.txt
brew install stockfish          # SF 18; or set STOCKFISH_PATH

# Provision an admin token + a key, then run:
export ADMIN_TOKEN=...           # guards key minting
export DATABASE_URL=postgres://...   # omit to use a local SQLite dev.db
uvicorn app.main:app --port 7860

# Mint a customer key:
python -m scripts.mint_key --email coach@club.com --quota 500
```

Deployment (Hugging Face Docker Space), architecture, and the build history live
in [PLAN.md](PLAN.md).

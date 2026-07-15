# Plan: Extract ExplainChess into a sellable API-product repo

## Context

ExplainChess predicts a per-player puzzle **solve-probability curve** + SHAP
explanation from a raw position (FEN + solution moves). The most sellable
packaging is a **usage-metered HTTP API** — coaches/platforms send puzzles and
get back difficulty curves — decoupled from the consumer web app.

The current backend (`api/main.py` in the training repo) is an *internal* API:
open CORS to the local frontend, **no auth, no metering, no rate limiting, no
billing**, single Stockfish behind a global lock, deployed on HF Spaces. This
plan stands up a **separate, self-contained repo** (`explainchess-api`) that
vendors in the minimal inference stack and adds the four things that make it
sellable: **API-key auth + metering, a batch endpoint, rate limiting, and
Stripe metered billing.**

Decisions already made by the user:
- **Vendor-copy** the pipeline into the new repo (self-contained, deploys
  independently) — not a pip dependency on the training repo.
- V1 scope = **all four**: batch endpoint, API keys + metering, rate limiting,
  Stripe billing.
- **Deploy on HF Spaces** (reuse today's setup; HF handles the 621 MB model
  natively, existing Dockerfile already works there).
- **Postgres from day one** for customers/keys/usage.
- **Manual-invoice first**: wire Stripe code in but keep it behind a flag;
  mint keys + meter locally + invoice pilots by hand until pricing is set.

## What the pipeline actually needs (verified against source)

`ExplainChessPipeline` (`src/pipeline.py`) is pure Python and already the
product entry point. Its runtime dependency closure is small:

**src/ modules to copy (inference-only subset):**
- `src/pipeline.py` — the entry point (`analyze_fen`, `solve_from_fen`,
  `get_calibration_puzzles`, `estimate_user_rating`, `close`)
- `src/stockfish/stockfish_features.py` — feature extraction (`extract_features_for_puzzle`, `decode_eval_columns`, `_pov_cp`)
- `src/stockfish/cup_data.py` — needed by `calibration.build_calibration_pool` (CSV path). See calibration note below.
- `src/stockfish/dataset.py` + `src/stockfish/feature_engineering.py` — **transitive** deps of `cup_data.py` (module-level imports). Their heavy code paths are never hit at inference, but they must be *importable*, so they must be copied.
- `src/tactical/tactical_features.py` — `TACTICAL_COLUMNS`, `extract_tactical_features`
- all `__init__.py` package markers for the copied subpackages
- `src/prediction_model/calibration.py` — `BANDS`, `RAPID_IDX`, `estimate_rating`, `load_calibration_pool_from_file`, `build_calibration_pool`
- `src/quality_pillar/flaw_agent.py` — `FlawAgent`, `DECISIVE_CP`, `UNIQUE_MARGIN` (only imports `_pov_cp` from stockfish_features — self-contained)
- `src/explainability/feature_descriptions.py` — `describe` (pipeline only uses this, NOT the SHAP-heavy `explain_curve.py`)
- `configs/config.yaml` — trimmed to the keys the pipeline reads: `stockfish.*` and `bigdata_cup.models_dir`

**Runtime artifacts to ship (NOT in git — see deploy note):**
- `results/models/perplayer_curve_lgbm.pkl` — **621 MB** (exceeds GitHub's 100 MB limit)
- `results/models/calibration_pool.json` — 46 KB (lets calibration run **data-free**)

**Calibration note:** `calibration.py` transitively imports `cup_data` which
loads the 2 GB training CSV — BUT `pipeline._load_calibration_pool()` prefers
`calibration_pool.json` when present (`load_calibration_pool_from_file`) and
only falls back to `build_calibration_pool` (needs the CSV) if the JSON is
missing. **Ship the JSON** → the whole API runs with zero dataset. Keep the
`cup_data` import working (copy the file) but the CSV path is never hit.

**System deps:** Stockfish **18** binary (pinned; apt's older build gives
different NNUE evals → skewed features). Resolved via `STOCKFISH_PATH` env var,
falling back to `config.yaml stockfish.path`.

**Python deps (inference subset of requirements.txt):** numpy, pandas, pyarrow,
python-chess, scikit-learn, lightgbm, shap, scipy, pyyaml, fastapi,
uvicorn[standard]. Drop tqdm (training-only). ADD the new API-layer deps:
`psycopg[binary]`/SQLAlchemy (Postgres), `stripe`, and a rate-limit lib
(`slowapi`).

**⚠ Pin exact `lightgbm` + `scikit-learn` versions** to whatever produced the
621 MB `.pkl`. The model is pickled; a version mismatch can fail to unpickle or
silently change predictions. Read the versions from the training repo's
environment before writing `requirements.txt`, and verify unpickling in step 1.

## New repo: `explainchess-api`

```
explainchess-api/
├── app/
│   ├── main.py              # FastAPI app: versioned /v1 routes, auth dep, rate limit
│   ├── auth.py              # API-key validation dependency (Bearer token)
│   ├── metering.py          # per-key usage counter + persistence
│   ├── billing.py           # Stripe metered-usage reporting
│   ├── ratelimit.py         # per-key throttle (slowapi or a small token bucket)
│   ├── schemas.py           # Pydantic request/response models (incl. batch)
│   └── db.py                # SQLite/Postgres: customers, api_keys, usage
├── engine/                  # vendored inference stack (the src/ subset above,
│   │                        #   re-rooted; update absolute imports src.* -> engine.*)
│   ├── pipeline.py
│   ├── stockfish/…  tactical/…  prediction_model/…  quality_pillar/…  explainability/…
├── models/                  # perplayer_curve_lgbm.pkl (via Git LFS / release asset),
│                            #   calibration_pool.json
├── config.yaml              # trimmed config
├── requirements.txt
├── Dockerfile               # adapt existing: install SF18, run uvicorn
├── README.md                # API quickstart + curl examples + pricing pointer
└── tests/                   # smoke tests for each endpoint incl. auth failure
```

**Import re-rooting:** the vendored modules use absolute `from src.…` imports.
Rename the package `src/` → `engine/` and rewrite `src.` → `engine.` across the
copied files (mechanical, ~7 files). Alternatively keep the package named `src`
to avoid edits — but `engine/` reads better for a product repo. Recommend the
rename.

## Build steps

1. **Scaffold repo + vendor the engine.** Copy the 7 modules + config, re-root
   imports to `engine.*`, add `requirements.txt`. Verify `python -c "from
   engine.pipeline import ExplainChessPipeline"` imports clean.

2. **Ship model artifacts.** `perplayer_curve_lgbm.pkl` is 621 MB → use **Git
   LFS** or a GitHub **release asset** downloaded at Docker build (mirror how
   the HF Space commits it into the repo). `calibration_pool.json` is small →
   commit directly. Confirm pipeline loads both.

3. **Datastore (`app/db.py`) — Postgres.** SQLAlchemy + `psycopg`; schema:
   `customers(id, email, stripe_customer_id, tier)`,
   `api_keys(key_hash, customer_id, active, created_at)`,
   `usage(key_hash, day, analyze_calls, batch_puzzles)`. Store **hashed** keys.
   Provision a managed Postgres (e.g. Neon/Supabase/Railway) and pass its URL as
   `DATABASE_URL`; HF Spaces has ephemeral disk, so the DB must be external.

4. **Auth (`app/auth.py`).** FastAPI dependency: read `Authorization: Bearer
   <key>`, hash, look up active key → attach `customer_id` to request. 401 on
   missing/invalid. A small `admin`-guarded `/v1/keys` endpoint (or a CLI
   script) to mint keys for pilot customers.

5. **Metering (`app/metering.py`).** After each successful `/v1/analyze` (and
   per-puzzle in batch), increment the `usage` row for (key, today). This is the
   source of truth for both tier limits and Stripe.

6. **Endpoints (`app/main.py`, versioned under `/v1`):**
   - `POST /v1/analyze` — wraps `pipeline.analyze_fen`; requires moves (422 if
     empty, matching existing behavior).
   - `POST /v1/analyze/batch` — `{puzzles:[{fen,moves},…]}` → list of results
     (or per-item `{error}`); loop under the Stockfish lock. **Cap batch size**
     (e.g. 100) and meter per-puzzle. This is the headline coach/platform value.
   - `POST /v1/solve` — wraps `solve_from_fen`.
   - `GET  /v1/calibration/puzzles`, `POST /v1/calibration/estimate` — optional
     for API buyers; keep for parity, low cost.
   - `GET  /health` — unauthenticated liveness (pipeline_ready).
   - `/docs` (FastAPI OpenAPI) is the API reference — ships free.

7. **Rate limiting (`app/ratelimit.py`).** Two concerns: (a) fairness — one
   Stockfish instance behind a global lock, so serialize + throttle per key
   (e.g. `slowapi` N req/min per key) so one customer can't starve others; (b)
   tier caps — reject over daily quota from the `usage` table (429). Consider a
   small pool of Stockfish processes later if throughput is the bottleneck.

8. **Billing (`app/billing.py`) — wired but OFF for V1.** Stripe metered
   subscription: map `customer.stripe_customer_id` → a metered subscription item;
   report usage (daily aggregate from the `usage` table). **Gate the whole thing
   behind a `BILLING_ENABLED` flag (default off).** V1 pilots: mint keys, meter
   in Postgres, invoice by hand from the `usage` table. Code ships so flipping
   the flag later is the only step to go live.

9. **Docker + deploy (HF Spaces).** Adapt the existing Dockerfile (already
   installs SF18 via the pinned GitHub release, non-root, uvicorn, port 7860).
   Changes: copy `app/` + `engine/` + `models/` + `config.yaml` (NOT the `.bak*`
   pkls); run `uvicorn app.main:app`. Commit the 621 MB `.pkl` into the Space
   repo (HF handles large files). Set Space **secrets**: `DATABASE_URL` (external
   Postgres), `STRIPE_API_KEY`, `BILLING_ENABLED=false`, `ADMIN_TOKEN` (key
   minting). Tighten `ALLOW_ORIGINS` — an API doesn't need open CORS.

## Verification

- **Import smoke:** `python -c "from engine.pipeline import ExplainChessPipeline; p=ExplainChessPipeline('config.yaml'); print(p.analyze_fen('<FEN>','e2e4 e7e5').keys())"` returns `dict_keys(['difficulty','quality','explanation','input_interpretation','diagnostics'])`.
- **Endpoint tests (`tests/`):**
  - `/v1/analyze` with a valid key → 200 + full response; with no/invalid key → 401; with empty moves → 422.
  - `/v1/analyze/batch` with 3 puzzles → list of 3; over-cap → 422; meters 3 puzzles.
  - Rate limit: fire > limit rapidly for one key → 429; a second key unaffected.
  - Metering: after N calls, `usage` row shows N; billing job reports N to Stripe (test mode).
- **End-to-end curl** in README against the running container (`docker build` +
  `docker run`, hit `/health` then `/v1/analyze` with a test key).
- **Model parity:** confirm the vendored pipeline gives identical
  `difficulty.label` / curve to the training repo for a known FEN (guards the
  import re-rooting + artifact copy).

## Confirmed decisions

- **Deploy:** HF Spaces (reuse existing Dockerfile; 621 MB model committed into
  the Space repo). Postgres must be **external** (Neon/Supabase/Railway) — HF
  disk is ephemeral.
- **Datastore:** Postgres from day one (SQLAlchemy + psycopg).
- **Billing:** Stripe code wired in but **behind `BILLING_ENABLED` flag (off for
  V1)**; pilots run on manual invoicing from the `usage` table.

"""Runtime configuration for the ExplainChess API.

All operational settings come from environment variables so the same image runs
locally and on Hugging Face Spaces (where these are set as Space secrets). The
one exception is the model/engine config, which stays in `config.yaml` next to
the vendored pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of this app/ package. Used to resolve config.yaml and the
# relative `models_dir` inside it, so the service works regardless of the cwd
# uvicorn is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(REPO_ROOT / "config.yaml")


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# --- Database ---------------------------------------------------------------
# External Postgres (HF Spaces disk is ephemeral). Falls back to a local SQLite
# file for dev / tests so you can run without provisioning Postgres.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", f"sqlite:///{REPO_ROOT / 'dev.db'}"
)

# --- Auth -------------------------------------------------------------------
# Guards the admin-only key-minting endpoint. Required in any real deployment.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# --- Rate limiting / quotas -------------------------------------------------
# Per-key requests/minute (fairness: one shared Stockfish behind a lock).
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
# Max puzzles accepted in a single /v1/analyze/batch call.
MAX_BATCH = int(os.environ.get("MAX_BATCH", "100"))

# --- Billing ----------------------------------------------------------------
# V1: OFF. Stripe code is wired but only reports usage when this is true.
BILLING_ENABLED = _bool("BILLING_ENABLED", False)
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")

# --- Engine -----------------------------------------------------------------
# STOCKFISH_PATH is read directly by the vendored pipeline; kept here for docs.
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "")

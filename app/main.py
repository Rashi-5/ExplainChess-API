"""ExplainChess API — the sellable HTTP surface.

Wraps the vendored ExplainChessPipeline behind API-key auth, per-key metering
and rate limiting, and (flag-gated) Stripe billing. The pipeline (Stockfish +
LightGBM) loads once at startup and is shared across requests behind a lock,
since Stockfish is single-threaded.

Routes are versioned under /v1 so internals can change without breaking
customers. Interactive reference docs are served at /docs (OpenAPI).
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy.orm import Session

from app import billing
from app.auth import Principal, get_session, require_admin, require_key
from app.config import CONFIG_PATH, MAX_BATCH, REPO_ROOT
from app.db import Customer, create_customer_with_key, init_db
from app.metering import check_quota, record_usage
from app.ratelimit import limiter
from app.schemas import (
    AnalyzeRequest, BatchAnalyzeRequest, BatchAnalyzeResponse, BatchItemResult,
    CalibrationEstimateRequest, MintKeyRequest, MintKeyResponse, SolveRequest,
)

# --- pipeline singleton -----------------------------------------------------
_pipeline = None
_lock = threading.Lock()        # guards Stockfish (single-threaded)
_calib_lock = threading.Lock()  # guards lazy calibration-pool init


def get_pipeline():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    init_db()
    # Resolve the model dir relative to repo root regardless of launch cwd.
    from engine.pipeline import ExplainChessPipeline

    import os
    os.chdir(REPO_ROOT)  # so config.yaml's relative models_dir resolves
    _pipeline = ExplainChessPipeline(config_path=CONFIG_PATH, load_calibration=False)
    yield
    _pipeline.close()
    _pipeline = None


app = FastAPI(
    title="ExplainChess API",
    version="1.0.0",
    description="Per-player puzzle difficulty curves + explanations, as a metered API.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# --- health (unauthenticated) ----------------------------------------------
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


# --- admin: mint keys -------------------------------------------------------
@app.post("/v1/keys", response_model=MintKeyResponse, tags=["admin"],
          dependencies=[Depends(require_admin)])
def mint_key(req: MintKeyRequest, session: Session = Depends(get_session)):
    customer, raw = create_customer_with_key(
        session, email=req.email, tier=req.tier, daily_quota=req.daily_quota
    )
    return MintKeyResponse(customer_id=customer.id, email=customer.email, api_key=raw)


# --- analyze ----------------------------------------------------------------
def _run_analyze(fen: str, moves: str) -> dict:
    moves = (moves or "").strip()
    if not moves:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Solution moves are required. Provide space-separated UCI moves "
                "(opponent's setup move first, then the solution), e.g. 'e2e4 d7d5'."
            ),
        )
    pipe = get_pipeline()
    with _lock:
        try:
            return pipe.analyze_fen(fen, moves)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=str(e))


@app.post("/v1/analyze", tags=["analysis"])
def analyze(
    req: AnalyzeRequest,
    principal: Principal = Depends(require_key),
    session: Session = Depends(get_session),
):
    check_quota(session, principal.key_hash, principal.customer, cost=1)
    result = _run_analyze(req.fen, req.moves)
    record_usage(session, principal.key_hash, analyze_calls=1)
    billing.report_usage(principal.customer, 1)
    return result


@app.post("/v1/analyze/batch", response_model=BatchAnalyzeResponse, tags=["analysis"])
def analyze_batch(
    req: BatchAnalyzeRequest,
    principal: Principal = Depends(require_key),
    session: Session = Depends(get_session),
):
    n = len(req.puzzles)
    if n > MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"Batch too large: {n} puzzles (max {MAX_BATCH}).",
        )
    # Quota is charged per puzzle; check up front for the whole batch.
    check_quota(session, principal.key_hash, principal.customer, cost=n)

    results: list[BatchItemResult] = []
    for i, p in enumerate(req.puzzles):
        try:
            results.append(BatchItemResult(index=i, result=_run_analyze(p.fen, p.moves)))
        except HTTPException as e:
            results.append(BatchItemResult(index=i, error=str(e.detail)))

    # Meter every puzzle we accepted for processing (success or per-item error),
    # matching how compute was actually spent.
    record_usage(session, principal.key_hash, batch_puzzles=n)
    billing.report_usage(principal.customer, n)
    return BatchAnalyzeResponse(results=results)


# --- solve ------------------------------------------------------------------
@app.post("/v1/solve", tags=["analysis"])
def solve(
    req: SolveRequest,
    principal: Principal = Depends(require_key),
    session: Session = Depends(get_session),
):
    check_quota(session, principal.key_hash, principal.customer, cost=1)
    pipe = get_pipeline()
    with _lock:
        try:
            result = pipe.solve_from_fen(req.fen, plies=req.plies)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))
    record_usage(session, principal.key_hash, analyze_calls=1)
    billing.report_usage(principal.customer, 1)
    return result


# --- calibration ------------------------------------------------------------
@app.get("/v1/calibration/puzzles", tags=["calibration"])
def calibration_puzzles(n: int = 15, principal: Principal = Depends(require_key)):
    pipe = get_pipeline()
    with _calib_lock:
        try:
            puzzles = pipe.get_calibration_puzzles(n=n)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))
    return {"puzzles": puzzles}


@app.post("/v1/calibration/estimate", tags=["calibration"])
def calibration_estimate(
    req: CalibrationEstimateRequest, principal: Principal = Depends(require_key)
):
    pipe = get_pipeline()
    try:
        return pipe.estimate_user_rating(req.history)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e))

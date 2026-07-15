"""Request/response models for the public API.

The analyze/solve *response* shapes come straight from the vendored pipeline
(deeply nested, stable), so they're passed through as plain dicts rather than
re-modelled here — that keeps this layer from drifting out of sync with the
engine. Only the request bodies and the batch envelope are typed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Moves-field help, shared across analyze endpoints. Mirrors the pipeline's own
# Lichess-convention contract.
_MOVES_HELP = (
    "Required. Space-separated UCI moves in Lichess convention: opponent's setup "
    "move first, then the solution. If the FEN is already the solver's position, "
    "prefix with '0000' (e.g. '0000 d8h4 g2g3 h4e4'). The model relies on these; "
    "empty moves are rejected."
)


class AnalyzeRequest(BaseModel):
    fen: str = Field(..., description="Board position in FEN.")
    moves: str = Field(default="", description=_MOVES_HELP)


class BatchAnalyzeRequest(BaseModel):
    puzzles: list[AnalyzeRequest] = Field(
        ..., description="List of puzzles to analyze in one call.", min_length=1
    )


class BatchItemResult(BaseModel):
    """One result slot in a batch response. Exactly one of `result`/`error` is set,
    so a single bad puzzle doesn't fail the whole batch."""
    index: int
    result: dict[str, Any] | None = None
    error: str | None = None


class BatchAnalyzeResponse(BaseModel):
    results: list[BatchItemResult]


class SolveRequest(BaseModel):
    fen: str = Field(..., description="Board position in FEN.")
    plies: int = Field(default=6, ge=1, le=12, description="Plies of the engine PV to play out.")


class CalibrationEstimateRequest(BaseModel):
    history: list[dict[str, Any]] = Field(
        ..., description='List of {"curve": [11 floats], "solved": 0|1}.'
    )


# --- admin ---

class MintKeyRequest(BaseModel):
    email: str
    tier: str = "pilot"
    daily_quota: int = Field(default=0, description="0 = unlimited.")


class MintKeyResponse(BaseModel):
    customer_id: int
    email: str
    api_key: str = Field(..., description="Raw key — shown ONCE. Store it now.")

"""
Tactical features inspired by the IEEE BigData 2024 Cup WINNER (Björkqvist).

The winning paper lists tactical/position features beyond plain engine evals:
number of checking moves, captures available, undefended pieces, mate threat,
recapture availability, mobility, etc. Several of these we did not yet compute.
This module adds the ones that need NO Stockfish — pure python-chess on the
FEN + solution moves — so they can be applied to the existing 700k extraction
without re-running the engine. Fully interpretable (SHAP-friendly).

Convention (matches stockfish_features.py): Moves[0] is the opponent's setup
move; the puzzle position is AFTER pushing it; Moves[1] is the solver's first
move. All position features are computed on that post-setup position, from the
solver's perspective.

Usage:
    from engine.tactical.tactical_features import build_tactical_features_df
    tac_df = build_tactical_features_df(df)   # df has FEN, Moves
"""

from __future__ import annotations

import logging

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

TACTICAL_COLUMNS: list[str] = [
    "num_legal_moves",
    "num_checking_moves",
    "num_captures_available",
    "num_undefended_own",
    "num_undefended_opp",
    "is_recapture",
    "first_move_is_capture",
    "material_imbalance_abs",
    # added to match the winner's feature list
    "side_to_move",       # 1 = white to move (solver), 0 = black
    "in_check_at_start",  # solver is already in check at the puzzle position
    "num_attacked_own",   # solver's pieces currently attacked by opponent (hanging-ish)
    # removed: first_move_gives_check, can_interpose_check, mate_in_1_available (~0 SHAP)
]

_PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _undefended(board: chess.Board, color: chess.Color) -> int:
    """Count `color`'s non-king pieces with no friendly defender."""
    n = 0
    for sq, pc in board.piece_map().items():
        if pc.color == color and pc.piece_type != chess.KING:
            if not board.attackers(color, sq):
                n += 1
    return n


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(_PIECE_VAL[pc.piece_type] for pc in board.piece_map().values() if pc.color == color)


def extract_tactical_features(fen: str, moves_str: str) -> dict[str, float]:
    """Compute tactical features for one puzzle (post-setup position, solver POV)."""
    moves = moves_str.strip().split()
    board = chess.Board(fen)

    if len(moves) >= 2 and moves[0] != "0000":
        # Lichess convention: moves[0] = opponent setup move
        setup = chess.Move.from_uci(moves[0])
        setup_was_capture = board.is_capture(setup)
        setup_to_sq = setup.to_square
        board.push(setup)
    else:
        # FEN-only mode: FEN is already the solver's position
        setup_was_capture = False
        setup_to_sq = None

    solver = board.turn
    legal = list(board.legal_moves)

    num_legal = len(legal)
    num_checks = sum(1 for m in legal if board.gives_check(m))
    num_caps = sum(1 for m in legal if board.is_capture(m))

    # recapture available: opponent's setup move was a capture, and the solver
    # can capture back on that square
    is_recap = 0
    if setup_was_capture and setup_to_sq is not None:
        is_recap = int(any(m.to_square == setup_to_sq and board.is_capture(m) for m in legal))

    sol = chess.Move.from_uci(moves[1]) if len(moves) >= 2 and moves[1] != "0000" else None
    first_cap = int(board.is_capture(sol)) if sol is not None and sol in legal else 0

    und_own = _undefended(board, solver)
    und_opp = _undefended(board, not solver)
    mat_imb = abs(_material(board, chess.WHITE) - _material(board, chess.BLACK))

    side = 1.0 if solver == chess.WHITE else 0.0
    in_check_start = float(board.is_check())

    attacked_own = sum(
        1 for sq, pc in board.piece_map().items()
        if pc.color == solver and board.attackers(not solver, sq)
    )

    return {
        "num_legal_moves": float(num_legal),
        "num_checking_moves": float(num_checks),
        "num_captures_available": float(num_caps),
        "num_undefended_own": float(und_own),
        "num_undefended_opp": float(und_opp),
        "is_recapture": float(is_recap),
        "first_move_is_capture": float(first_cap),
        "material_imbalance_abs": float(mat_imb),
        "side_to_move": side,
        "in_check_at_start": in_check_start,
        "num_attacked_own": float(attacked_own),
    }


def build_tactical_features_df(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """Compute tactical features for every puzzle. Same-indexed as df (FEN, Moves)."""
    records: list[dict] = []
    it = tqdm(df.itertuples(), total=len(df), desc="Tactical features") if show_progress else df.itertuples()
    for row in it:
        try:
            records.append(extract_tactical_features(row.FEN, row.Moves))
        except Exception as exc:
            logger.debug("tactical fail on %s: %s", getattr(row, "PuzzleId", "?"), exc)
            records.append({c: float("nan") for c in TACTICAL_COLUMNS})
    return pd.DataFrame(records, index=df.index)


if __name__ == "__main__":
    # smoke test on a few hand-made puzzles
    samples = [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1 e8g8 d2d3 d7d6"),
        ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8 g8g7 e8e7 g7g6"),
    ]
    for fen, mv in samples:
        f = extract_tactical_features(fen, mv)
        print(fen[:30], "->", {k: round(v, 1) for k, v in f.items()})

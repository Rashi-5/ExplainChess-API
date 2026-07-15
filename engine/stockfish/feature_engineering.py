"""
Phase 1a + 1c: Board tensor and move-derived features.

Board tensor: FEN → numpy array of shape (12, 8, 8)
  Channels 0-5:  white  P N B R Q K
  Channels 6-11: black  p n b r q k

Move features (no Stockfish required):
  first_move_is_sacrifice  — 1 if the first move loses material
"""

from __future__ import annotations

import logging

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Piece-type → channel index (white=0..5, black=6..11)
_PIECE_TO_CHANNEL: dict[tuple[chess.Color, chess.PieceType], int] = {
    (chess.WHITE, chess.PAWN):   0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK):   3,
    (chess.WHITE, chess.QUEEN):  4,
    (chess.WHITE, chess.KING):   5,
    (chess.BLACK, chess.PAWN):   6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK):   9,
    (chess.BLACK, chess.QUEEN):  10,
    (chess.BLACK, chess.KING):   11,
}

# Centipawn values for material calculation
_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}


def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Convert a FEN string to a (12, 8, 8) float32 numpy array.

    The board is oriented with rank 1 at index 0 (consistent with python-chess
    square ordering where a1=0, h8=63).

    Returns:
        Array of shape (12, 8, 8), dtype float32.
    """
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        channel = _PIECE_TO_CHANNEL[(piece.color, piece.piece_type)]
        rank = chess.square_rank(square)   # 0..7
        file = chess.square_file(square)   # 0..7
        tensor[channel, rank, file] = 1.0

    return tensor


def extract_move_features(fen: str, moves_str: str) -> dict[str, float]:
    """
    Extract move-derived features that require no Stockfish engine.

    Args:
        fen:       Starting position FEN.
        moves_str: Space-separated UCI move sequence (the puzzle solution).

    Returns:
        Dict with keys: first_move_is_sacrifice.
    """
    moves = moves_str.strip().split()
    first_move_is_sacrifice = 0.0

    if len(moves) >= 2:
        board = chess.Board(fen)
        try:
            # Push the opponent's setup move first (#1: same fix as stockfish_features)
            setup = chess.Move.from_uci(moves[0])
            if board.is_legal(setup):
                board.push(setup)
                solver_move = chess.Move.from_uci(moves[1])
                if board.is_legal(solver_move):
                    material_before = _material_balance(board, board.turn)
                    board.push(solver_move)
                    material_after = _material_balance(board, not board.turn)
                    if material_after < material_before - 50:   # lost >50 cp
                        first_move_is_sacrifice = 1.0
        except (ValueError, AssertionError):
            pass

    return {
        "first_move_is_sacrifice": first_move_is_sacrifice,
    }


def _material_balance(board: chess.Board, color: chess.Color) -> int:
    """Net material for `color` minus opponent (in centipawns)."""
    score = 0
    for piece_type, value in _PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
        score -= len(board.pieces(piece_type, not color)) * value
    return score


def build_move_features_df(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """
    Compute move-derived features for every puzzle in df.

    Args:
        df:            DataFrame with columns FEN, Moves.
        show_progress: Show tqdm progress bar.

    Returns:
        DataFrame indexed the same as df with move feature columns.
    """
    records: list[dict] = []
    iterator = tqdm(df.itertuples(), total=len(df), desc="Move features") if show_progress else df.itertuples()

    for row in iterator:
        try:
            feats = extract_move_features(row.FEN, row.Moves)
        except Exception as exc:
            logger.debug("Error on PuzzleId %s: %s", getattr(row, "PuzzleId", "?"), exc)
            feats = {"first_move_is_sacrifice": float("nan")}
        records.append(feats)

    return pd.DataFrame(records, index=df.index)


def build_board_tensors(df: pd.DataFrame, show_progress: bool = True) -> np.ndarray:
    """
    Build board tensors for all puzzles in df.

    The FEN in the Lichess CSV is the position *before* the opponent's setup
    move (Moves[0]). We push that move first so the CNN sees the actual puzzle
    position — the same position Stockfish analyses.

    Returns:
        Array of shape (N, 12, 8, 8), dtype float32.
    """
    n = len(df)
    tensors = np.zeros((n, 12, 8, 8), dtype=np.float32)
    iterator = (
        tqdm(enumerate(zip(df["FEN"], df["Moves"])), total=n, desc="Board tensors")
        if show_progress
        else enumerate(zip(df["FEN"], df["Moves"]))
    )

    for i, (fen, moves_str) in iterator:
        try:
            board = chess.Board(fen)
            setup = chess.Move.from_uci(moves_str.strip().split()[0])
            if board.is_legal(setup):
                board.push(setup)
            tensors[i] = fen_to_tensor(board.fen())
        except Exception as exc:
            logger.debug("Bad FEN/move at index %d: %s", i, exc)

    return tensors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import sys
    sys.path.insert(0, ".")
    from engine.data_loader import load_clean_puzzles

    print("Loading 5 000 puzzles for a quick smoke-test ...")
    df = load_clean_puzzles(sample=5000)

    print("Building board tensors ...")
    tensors = build_board_tensors(df)
    print(f"  tensor array shape: {tensors.shape}")
    print(f"  example tensor sum (should equal piece count): {tensors[0].sum():.0f}")

    print("Extracting move features ...")
    move_df = build_move_features_df(df)
    print(move_df.describe())
    print("\nPhase 1a + 1c complete.")

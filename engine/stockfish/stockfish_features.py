"""
Stockfish feature-extraction core.

Fixes applied (numbered against the review):
  #1  Correct Lichess indexing: push Moves[0] (opponent setup move) first;
      the solver's move is Moves[1]; all engine features are computed on the
      post-setup position from the SOLVER's point of view.
  #2  One persistent engine per worker (Pool initializer), not one per puzzle.
  #3  Safe two-sided mobility via a board copy (no raw board.turn mutation).
  #4  Single 'num_solution_moves' = count of solver plies (no duplicate).
  #5  is_mate flag + graded mate score that preserves mate-in-N ordering.
  #7  Failure counter returned per row so you can report the NaN rate.
  #9  engine.configure(Threads=1) for reproducibility.

Novel additions vs literature:
  - Solver-POV evals (not white-POV) for correct Black puzzle handling.
  - multi-depth evals (d1, d3, d5) alongside main depth: the delta between
    shallow and deep eval is a direct measure of tactical sharpness.
  - best_move_depth: depth at which SF first agrees with the solution.
  - Dual candidate thresholds (50cp + 100cp).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

_PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

# All numeric feature keys — used to NaN-fill on failure (#7).
_FEATURE_KEYS = [
    "cp_eval", "cp_eval_d1", "eval_delta",
    "cp_eval_after", "eval_swing", "best_move_depth",
    "num_candidate_moves", "num_candidate_moves_50cp", "num_solution_moves",
    "material_balance", "total_material", "num_pieces",
    "king_safety_white", "king_safety_black", "is_check", "is_capture",
    "is_promotion", "is_mate", "pawn_structure_score", "mobility_diff",
    # line-level features (walk the full solution sequence)
    "only_move_count", "only_move_ratio", "mean_branching_factor",
    "quiet_move_count", "solution_eval_range", "final_eval",
]

# ── scoring helpers ──────────────────────────────────────────────────────────

def _pov_cp(info: dict[str, Any], pov_color: chess.Color) -> float:
    """Centipawns from pov_color's perspective with graded mate scoring (#5).

    Normal evals clamped to +-3000. Mates map above that band while preserving
    mate-in-N ordering: mate-in-1 -> 9900, mate-in-2 -> 9800, etc.
    """
    score = info["score"].pov(pov_color)
    if score.is_mate():
        n = abs(score.mate())
        mag = 10000.0 - min(n, 50) * 100.0
        # score.mate() collapses to plain int 0 at the mate-just-delivered/received
        # boundary, losing the +0/-0 sign distinction str(score) shows (e.g. "#+0"
        # vs "#-0") — compare the Score object itself against Cp(0) instead, which
        # stays sign-correct even when mate() rounds to zero.
        return mag if score >= chess.engine.Cp(0) else -mag
    cp = float(score.score())
    return max(-3000.0, min(3000.0, cp))


def _walk_solution_line(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    solver: chess.Color,
    line_moves: list[chess.Move],
    depth_line: int,
    game: object,
) -> dict[str, float]:
    """Walk the full solution line, analysing only the solver's plies.

    `board` must be at the post-setup position (solver to move) and is NOT
    mutated (operates on a copy). `line_moves` is the full remaining ply
    sequence — solver and opponent moves interleaved, in order. Opponent plies
    are pushed without analysis.

    A single-PV eval is taken at each solver ply (multipv was dropped: the
    only-move / branching features it powered are ~constant on Lichess puzzles,
    since every solver ply is the unique winning move by construction). The
    surviving aggregates — quiet-move count, eval range and final eval across
    the whole line — are the ones that empirically separate hard from easy.
    """
    b = board.copy()
    quiet_moves = 0
    evals: list[float] = []

    for mv in line_moves:
        if b.turn == solver:
            info = engine.analyse(b, chess.engine.Limit(depth=depth_line), game=game)
            evals.append(_pov_cp(info, solver))
            if not (b.gives_check(mv) or b.is_capture(mv) or mv.promotion):
                quiet_moves += 1
        b.push(mv)

    return {
        "quiet_move_count": float(quiet_moves),
        "solution_eval_range": (max(evals) - min(evals)) if evals else float("nan"),
        "final_eval": evals[-1] if evals else float("nan"),
    }


def decode_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Disentangle the graded-mate encoding from `_pov_cp` into clean channels.

    The extractor (#5) packs mate-in-N into the centipawn axis as a sentinel
    band: |value| > 3000 means a mate, with magnitude 10000 - 100*n preserving
    mate-in-N ordering. Feeding that raw into a model corrupts the continuous
    eval scale (a +9900 "mate" reads as 30x a +300 real eval) and makes
    eval_delta/eval_swing arithmetic across the 3000->9900 gap meaningless.

    This splits each raw eval column into two coherent features:
      <col>_cp        real centipawns, clipped to +-2000; mate rows pinned to
                      the signed clip ceiling so the channel stays monotonic.
      <col>_mate_ply  signed mate distance: +n if the POV side delivers mate
                      in n plies, -n if it is being mated, 0 if no mate.
                      (n=0, i.e. raw -10000, means already-terminal => -0.5 ply
                      sentinel so "mated immediately" is distinct from "no mate".)

    eval_delta and eval_swing are recomputed from the *_cp parts only, so the
    arithmetic stays on a single coherent scale; the mate_ply columns carry the
    forcing signal separately.

    Operates on a copy; returns a new DataFrame with the extra columns added.
    Recovery is exact (verified: no real eval lands in (3000, 9000), and
    decoded n is always an integer in [0, 50]).
    """
    out = df.copy()
    raw_eval_cols = ["cp_eval", "cp_eval_d1", "cp_eval_after"]
    CLIP = 2000.0
    MATE_THRESH = 3000.0

    for col in raw_eval_cols:
        v = out[col]
        is_mate_row = v.abs() > MATE_THRESH
        # decode signed mate-in-n; sign follows the sentinel sign (solver vs mated)
        n = (10000.0 - v.abs()) / 100.0          # exact integer plies, 0..50
        sign = np.sign(v)
        mate_ply = np.where(is_mate_row, sign * np.where(n == 0, 0.5, n), 0.0)
        # continuous channel: real cp clipped; mate rows pinned to signed ceiling
        cp = np.where(is_mate_row, sign * CLIP, v.clip(-CLIP, CLIP))
        out[f"{col}_cp"] = cp.astype(np.float32)
        out[f"{col}_mate_ply"] = mate_ply.astype(np.float32)

    # recompute deltas/swings on the clipped continuous parts only
    out["eval_delta"] = (out["cp_eval_cp"] - out["cp_eval_d1_cp"]).abs().astype(np.float32)
    out["eval_swing"] = (out["cp_eval_after_cp"] - out["cp_eval_cp"]).astype(np.float32)
    return out


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(pt, color)) * val for pt, val in _PIECE_VALUES.items()
    )


def _king_attackers(board: chess.Board, color: chess.Color) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    ring = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]
    return sum(len(board.attackers(not color, sq)) for sq in chess.SquareSet(ring))


def _isolated_doubled_pawns(board: chess.Board) -> int:
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        files: dict[int, int] = {}
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            files[f] = files.get(f, 0) + 1
        for f, cnt in files.items():
            if cnt > 1:
                score += cnt - 1                          # doubled
            if (f - 1) not in files and (f + 1) not in files:
                score += 1                                # isolated
    return score


def _mobility_for(board: chess.Board, color: chess.Color) -> int:
    """Legal-move count for color without mutating the live board (#3)."""
    if board.turn == color:
        return board.legal_moves.count()
    b = board.copy(stack=False)
    b.turn = color
    b.ep_square = None
    return b.legal_moves.count()


# ── per-puzzle extraction ────────────────────────────────────────────────────

def extract_features_for_puzzle(
    engine: chess.engine.SimpleEngine,
    puzzle_id: str,
    fen: str,
    moves_str: str,
    depth: int,
    depth_candidates: int,
    multipv: int,
    depth_line: int = 10,
    only_move_margin: float = 150.0,
) -> dict[str, Any]:
    """Extract features for one puzzle using an already-open engine (#2).

    On error, numeric fields are NaN and extract_ok is False (#7).
    """
    result: dict[str, Any] = {"PuzzleId": puzzle_id, "extract_ok": True}
    moves = moves_str.strip().split()

    # A fresh sentinel per call: passed as `game=` to every engine.analyse()
    # below so python-chess sends `ucinewgame` before this puzzle's searches,
    # clearing Stockfish's hash table. Without this, puzzle_id is often a
    # constant ("live" in the pipeline), so the engine.game != game check
    # never trips past the first call ever made on a long-lived engine
    # process — later puzzles' searches get polluted by hash-table entries
    # left over from earlier, unrelated positions (nondeterministic-looking
    # eval drift under sustained use, e.g. a long-running deployed server).
    game = object()

    try:
        board = chess.Board(fen)

        if len(moves) >= 2 and moves[0] == "0000":
            # FEN-is-already-solver mode: no opponent setup move (e.g. an
            # engine-solved position from the board editor). moves[1] is the
            # solver's first move; the rest of the line follows from this FEN.
            solver = board.turn
            sol_move = chess.Move.from_uci(moves[1])
            if not board.is_legal(sol_move):
                raise ValueError(f"illegal solution move {moves[1]}")
        elif len(moves) >= 2:
            # Lichess convention: moves[0] = opponent setup move, moves[1] = solver's first move
            setup = chess.Move.from_uci(moves[0])
            if not board.is_legal(setup):
                raise ValueError(f"illegal setup move {moves[0]}")
            board.push(setup)
            solver = board.turn
            sol_move = chess.Move.from_uci(moves[1])
            if not board.is_legal(sol_move):
                raise ValueError(f"illegal solution move {moves[1]}")
        else:
            # FEN-only mode: the FEN is already the solver's position; ask Stockfish for best move
            solver = board.turn
            info_best = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1, game=game)
            pv = info_best.get("pv") if isinstance(info_best, dict) else info_best[0].get("pv")
            if not pv:
                raise ValueError("Stockfish found no legal moves from this position")
            sol_move = pv[0]
            moves = ["0000", sol_move.uci()]  # synthesise moves list for downstream use

        # ── shallow eval (d1) then full-depth ────────────────────────────────
        # d1 vs full-depth delta = tactical sharpness: large delta means the
        # position needs deep search to evaluate correctly (harder puzzle).
        info_d1 = engine.analyse(board, chess.engine.Limit(depth=1), game=game)
        result["cp_eval_d1"] = _pov_cp(info_d1, solver)

        info = engine.analyse(board, chess.engine.Limit(depth=depth), game=game)
        cp_eval = _pov_cp(info, solver)
        result["cp_eval"] = cp_eval
        result["is_mate"] = int(info["score"].pov(solver).is_mate())

        # eval_delta: how much the eval changes from d1 to full depth
        result["eval_delta"] = abs(cp_eval - result["cp_eval_d1"])

        # best_move_depth: the shallowest depth at which SF's PV first agrees
        # with the solution move (1..depth), not a binary 1-vs-full-depth flag.
        # A binary encoding collapses "found at depth 2" and "found at depth 11"
        # into the same value, which misrepresents genuinely-easy tactics (e.g.
        # a forced-mate reply that's obvious to a human but needs a couple of
        # plies for SF to rank over quieter alternatives) as maximally hard.
        if info_d1.get("pv") and info_d1["pv"][0] == sol_move:
            result["best_move_depth"] = 1
        else:
            result["best_move_depth"] = depth
            for d in range(2, depth):
                info_d = engine.analyse(board, chess.engine.Limit(depth=d), game=game)
                pv_d = info_d.get("pv")
                if pv_d and pv_d[0] == sol_move:
                    result["best_move_depth"] = d
                    break

        # ── solution-move flags ───────────────────────────────────────────────
        result["is_capture"] = int(board.is_capture(sol_move))
        board_after = board.copy()
        board_after.push(sol_move)
        result["is_check"] = int(board_after.is_check())
        info_after = engine.analyse(board_after, chess.engine.Limit(depth=depth), game=game)
        cp_eval_after = _pov_cp(info_after, solver)
        result["cp_eval_after"] = cp_eval_after
        result["eval_swing"] = cp_eval_after - cp_eval

        # promotion anywhere in the solver's plies (odd indices = solver moves)
        result["is_promotion"] = int(any(
            chess.Move.from_uci(u).promotion for u in moves[1::2]
        ))

        # #4: solver plies only (no duplicate solution_length)
        result["num_solution_moves"] = len(moves[1::2])

        # ── candidate moves at 50cp and 100cp (novel: dual threshold) ────────
        multipv_info = engine.analyse(
            board, chess.engine.Limit(depth=depth_candidates), multipv=multipv, game=game
        )
        if not isinstance(multipv_info, list):
            multipv_info = [multipv_info]
        cps = [_pov_cp(mi, solver) for mi in multipv_info]
        best_cp = max(cps)
        result["num_candidate_moves"]      = sum(1 for c in cps if best_cp - c <= 100)
        result["num_candidate_moves_50cp"] = sum(1 for c in cps if best_cp - c <= 50)

        # ── static position features ──────────────────────────────────────────
        white_mat = _material(board, chess.WHITE)
        black_mat = _material(board, chess.BLACK)
        result["material_balance"]    = white_mat - black_mat
        result["total_material"]      = white_mat + black_mat
        result["num_pieces"]          = len(board.piece_map())
        result["king_safety_white"]   = _king_attackers(board, chess.WHITE)
        result["king_safety_black"]   = _king_attackers(board, chess.BLACK)
        result["pawn_structure_score"] = _isolated_doubled_pawns(board)
        result["mobility_diff"] = (
            _mobility_for(board, chess.WHITE) - _mobility_for(board, chess.BLACK)
        )

        # ── line-level features: walk the full solution sequence ──────────────
        # `board` is still at the post-setup position (solver to move) — the
        line_moves = [chess.Move.from_uci(u) for u in moves[1:]]
        line_feats = _walk_solution_line(
            engine, board, solver, line_moves, depth_line=depth_line, game=game,
        )
        result.update(line_feats)

    except Exception as exc:
        logger.warning("Feature extraction failed for %s: %s", puzzle_id, exc)  # #7
        result["extract_ok"] = False
        for k in _FEATURE_KEYS:
            result.setdefault(k, float("nan"))

    return result


# ── worker with a persistent engine (#2, #9) ─────────────────────────────────

_ENGINE: chess.engine.SimpleEngine | None = None


def _init_worker(engine_path: str) -> None:
    
    global _ENGINE
    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _ENGINE.configure({"Threads": 1})        # #9: deterministic, no oversubscription


def _worker(args: tuple) -> dict[str, Any]:
    puzzle_id, fen, moves_str, depth, depth_candidates, multipv, depth_line, only_move_margin = args
    assert _ENGINE is not None, "engine not initialised"
    return extract_features_for_puzzle(
        _ENGINE, puzzle_id, fen, moves_str, depth, depth_candidates, multipv,
        depth_line=depth_line, only_move_margin=only_move_margin,
    )


# ── batch processing ─────────────────────────────────────────────────────────

def process_batch(
    batch_df: pd.DataFrame,
    batch_idx: int,
    out_dir: Path,
    engine_path: str,
    depth: int,
    depth_candidates: int,
    multipv: int,
    num_workers: int,
    depth_line: int = 10,
    only_move_margin: float = 150.0,
) -> int:
    """Process one batch, save to parquet, return count of failed rows (#7)."""
    tasks = [
        (row.PuzzleId, row.FEN, row.Moves, depth, depth_candidates, multipv,
         depth_line, only_move_margin)
        for row in batch_df.itertuples()
    ]

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(engine_path,),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(_worker, tasks, chunksize=50),
                total=len(tasks),
                desc=f"  Batch {batch_idx:04d}",
                leave=False,
            )
        )

    result_df = pd.DataFrame(results)
    n_failed = int((~result_df["extract_ok"]).sum())
    out_path = out_dir / f"batch_{batch_idx:04d}.parquet"
    result_df.to_parquet(out_path, index=False)
    logger.info(
        "Saved batch %04d -> %s (%d rows, %d failed)",
        batch_idx, out_path, len(result_df), n_failed,
    )
    return n_failed


# ── orchestration ─────────────────────────────────────────────────────────────

def run_extraction(
    data_csv: str,
    out_dir: str,
    engine_path: str,
    depth: int = 8,
    depth_candidates: int = 10,
    multipv: int = 5,
    batch_size: int = 10000,
    num_workers: int | None = None,
    sample: int | None = None,
    stratify: bool = False,
    rating_bins: list[int] | None = None,
    depth_line: int = 10,
    only_move_margin: float = 150.0,
) -> None:
    """Full pipeline: load CSV -> undersample -> process in batches -> parquet.

    Already-completed batches are skipped automatically (resume-safe).
    With --stratify, equal-per-bucket undersampling is used so all rating
    ranges are equally represented (justified for explainability research).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    logger.info("Using %d worker processes", num_workers)

    logger.info("Loading puzzle CSV ...")
    if stratify and sample:
        df_full = pd.read_csv(data_csv, usecols=["PuzzleId", "FEN", "Moves", "Rating"])
        bins = rating_bins or [800, 1200, 1600, 2000, 9999]
        df_full["_bucket"] = pd.cut(df_full["Rating"], bins=bins, labels=False)
        n_buckets = df_full["_bucket"].nunique()
        n_per_bucket = sample // n_buckets
        df = (
            df_full.groupby("_bucket", group_keys=False)
            .apply(lambda g: g.sample(n=min(n_per_bucket, len(g)), random_state=42))
            .drop(columns=["Rating", "_bucket"])
            .reset_index(drop=True)
        )
        logger.info(
            "Undersampled: %d puzzles, %d per bucket across %d buckets",
            len(df), n_per_bucket, n_buckets,
        )
    else:
        df = pd.read_csv(data_csv, usecols=["PuzzleId", "FEN", "Moves"], nrows=sample)
    logger.info("Total puzzles to process: %d", len(df))

    total_failed = 0
    num_batches = (len(df) + batch_size - 1) // batch_size
    logger.info("Batches: %d (size=%d)", num_batches, batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        out_file = out_path / f"batch_{batch_idx:04d}.parquet"
        if out_file.exists():
            logger.info("Skipping batch %04d (already done)", batch_idx)
            continue

        start = batch_idx * batch_size
        batch_df = df.iloc[start: start + batch_size]
        total_failed += process_batch(
            batch_df=batch_df,
            batch_idx=batch_idx,
            out_dir=out_path,
            engine_path=engine_path,
            depth=depth,
            depth_candidates=depth_candidates,
            multipv=multipv,
            num_workers=num_workers,
            depth_line=depth_line,
            only_move_margin=only_move_margin,
        )

    logger.info(
        "Extraction complete. Total failures: %d / %d (%.1f%%)",
        total_failed, len(df), 100 * total_failed / max(len(df), 1),
    )

    logger.info("Merging batches ...")
    parts = sorted(out_path.glob("batch_*.parquet"))
    merged = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    merged_path = out_path / "all_features.parquet"
    merged.to_parquet(merged_path, index=False)
    logger.info(
        "Merged -> %s  (%d rows, %d cols)", merged_path, len(merged), len(merged.columns)
    )


def load_stockfish_features(out_dir: str) -> pd.DataFrame:
    """Load merged feature file, or concat batches if merge not yet done."""
    out_path = Path(out_dir)
    merged = out_path / "all_features.parquet"
    if merged.exists():
        return pd.read_parquet(merged)
    parts = sorted(out_path.glob("batch_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {out_dir}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Extract Stockfish features for Lichess puzzles")
    parser.add_argument("--sample", type=int, default=None, help="Process only N puzzles")
    parser.add_argument("--stratify", action="store_true", help="Equal-per-bucket undersampling across rating bins")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_extraction(
        data_csv=cfg["paths"]["data_csv"],
        out_dir=cfg["paths"]["stockfish_features_dir"],
        engine_path=cfg["stockfish"]["path"],
        depth=cfg["stockfish"]["depth"],
        depth_candidates=cfg["stockfish"]["depth_candidates"],
        multipv=cfg["stockfish"]["multipv"],
        batch_size=cfg["stockfish"]["batch_size"],
        num_workers=cfg["stockfish"]["num_workers"],
        sample=args.sample,
        stratify=args.stratify,
        rating_bins=cfg["training"]["rating_bins"],
        depth_line=cfg["stockfish"].get("depth_line", 10),
        only_move_margin=cfg["stockfish"].get("only_move_margin", 150),
    )


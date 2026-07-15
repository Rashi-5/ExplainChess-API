"""
IEEE BigData 2024 Cup data loading + feature assembly.

Single source of truth so every experiment is measured on the SAME data and
the SAME fixed split. Competition metric is MSE; the official test labels are
secret, so we evaluate on a fixed held-out slice of the official train set and
clearly label it as such in the dissertation.

Feature groups (kept separate so we can ablate what drives the score):
  engine        — Stockfish + move features + decoded eval channels (explainable core)
  success_prob  — the 22 provided per-rating-band solve-probability columns
  curve         — engineered summaries of the success_prob curves

Usage:
    from engine.stockfish.cup_data import load_cup_frame, build_feature_groups, make_holdout
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.stockfish.dataset import FEATURE_COLUMNS
from engine.stockfish.feature_engineering import build_move_features_df
from engine.stockfish.stockfish_features import decode_eval_columns, load_stockfish_features

logger = logging.getLogger(__name__)

SUCCESS_PROB_COLS: list[str] = (
    [f"success_prob_rapid_{r}" for r in range(1050, 2051, 100)]
    + [f"success_prob_blitz_{r}" for r in range(1050, 2051, 100)]
)
_RATING_GRID = np.arange(1050, 2051, 100)  # the 11 rating points the probs are sampled at


def load_cup_frame(
    cfg: dict,
    sample: int | None = None,
    require_engine: bool = True,
    features_dir: str | None = None,
    with_tactical: bool = False,
) -> pd.DataFrame:
    """Load official train CSV, attach engine features, build all feature columns.

    Args:
        require_engine: if True, inner-join to puzzles that have Stockfish
            features (so engine columns are real, not median-filled). If False,
            keep all official rows (engine columns may be NaN where unavailable).
        features_dir: override the Stockfish features directory (else uses
            cfg paths). Lets a 700k extraction be used without editing config.
    """
    cup = cfg["bigdata_cup"]
    usecols = ["PuzzleId", "FEN", "Moves", "Rating"] + SUCCESS_PROB_COLS
    logger.info("Loading official Cup train CSV ...")
    df = pd.read_csv(cup["train_csv"], usecols=usecols).dropna(subset=["Rating"])

    sf = load_stockfish_features(features_dir or cfg["paths"]["stockfish_features_dir"])
    if require_engine:
        df = df.merge(sf, on="PuzzleId", how="inner")
        logger.info("Inner-joined to engine features: %d puzzles", len(df))
    else:
        df = df.merge(sf, on="PuzzleId", how="left")
        logger.info("Left-joined engine features: %d puzzles (%d with engine)",
                    len(df), df["cp_eval"].notna().sum())

    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=cfg["training"]["seed"]).reset_index(drop=True)

    logger.info("Building move features + decoding eval channels ...")
    move_df = build_move_features_df(df)
    df = pd.concat([df.reset_index(drop=True), move_df.reset_index(drop=True)], axis=1)
    df = decode_eval_columns(df)
    df = _add_curve_features(df)

    if with_tactical:
        from engine.tactical.tactical_features import build_tactical_features_df
        logger.info("Building tactical features (python-chess) ...")
        tac_df = build_tactical_features_df(df)
        df = pd.concat([df.reset_index(drop=True), tac_df.reset_index(drop=True)], axis=1)
    return df


def _add_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each puzzle's solve-probability curve into compact features.

    The 11-point curve per time control encodes how solve-rate falls with player
    rating; its shape (level, slope, where it crosses 0.5) is a strong difficulty
    signal. These complement the raw 22 columns for tree/linear models.
    """
    out = df.copy()
    for name in ("rapid", "blitz"):
        cols = [f"success_prob_{name}_{r}" for r in range(1050, 2051, 100)]
        B = out[cols].values
        out[f"{name}_prob_mean"] = B.mean(axis=1)
        out[f"{name}_prob_min"] = B.min(axis=1)
        out[f"{name}_prob_max"] = B.max(axis=1)
        out[f"{name}_prob_slope"] = B[:, -1] - B[:, 0]
        # interpolated rating where the curve crosses 0.5 (the puzzle's "even" rating)
        cross = np.full(len(B), 2050.0)
        below = B <= 0.5
        has = below.any(axis=1)
        first = np.argmax(below, axis=1)  # first True index (0 if none, guarded by `has`)
        cross[has] = _RATING_GRID[first[has]]
        out[f"{name}_prob_cross50"] = cross
    return out


CURVE_COLS: list[str] = [
    f"{n}_prob_{s}"
    for n in ("rapid", "blitz")
    for s in ("mean", "min", "max", "slope", "cross50")
]


def build_feature_groups(
    df: pd.DataFrame, train_idx: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Return the three feature-group matrices, NaN-filled with column medians.

    Leakage control: if ``train_idx`` is given, engine-feature medians are computed
    on the TRAINING ROWS ONLY and used to fill the whole column, so no hold-out
    information leaks into the imputation statistic. If omitted, medians fall back
    to the full-column median (legacy behaviour; kept for callers that have not yet
    passed the split).
    """
    eng = df[FEATURE_COLUMNS].copy()
    medians = eng.iloc[train_idx].median() if train_idx is not None else eng.median()
    eng = eng.fillna(medians)
    return {
        "engine": eng.values.astype(np.float32),
        "success_prob": df[SUCCESS_PROB_COLS].values.astype(np.float32),
        "curve": df[CURVE_COLS].values.astype(np.float32),
    }


def make_holdout(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed, seeded, rating-stratified train/holdout index split.

    Returns (idx_train, idx_holdout). Same split regardless of feature group so
    every experiment is comparable.
    """
    from sklearn.model_selection import train_test_split

    tcfg = cfg["training"]
    y = df["Rating"].values.astype(np.float32)
    strat = np.digitize(y, tcfg["rating_bins"])
    idx = np.arange(len(df))
    idx_train, idx_holdout = train_test_split(
        idx,
        test_size=cfg["bigdata_cup"]["holdout_size"],
        stratify=strat,
        random_state=tcfg["seed"],
    )
    return idx_train, idx_holdout

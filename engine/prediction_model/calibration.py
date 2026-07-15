"""
User skill calibration — estimate a user's rating from their solve/fail pattern
on a few puzzles, using the predicted per-player difficulty curves.

Principle (same idea as Lichess puzzle rating, run in reverse): the per-player
model predicts P(solve | user_rating) for each puzzle. Given which puzzles a user
solved/failed, find the rating R whose predicted solve-probabilities best explain
that pattern (maximum likelihood over the 11 rating bands).

Design choice: calibration puzzles are chosen to be the most SKILL-DISCRIMINATING
(steepest curves) — flat-curve puzzles carry no information about user skill.

Includes a self-validation: simulate users of known rating, check the estimator
recovers it. This is the proof-of-concept built into the module.

Usage:
    python -m src.perplayer.calibration --sample 8000 --n_calib 12
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import yaml

from engine.stockfish.cup_data import SUCCESS_PROB_COLS, build_feature_groups, load_cup_frame, make_holdout
from engine.tactical.tactical_features import TACTICAL_COLUMNS

warnings.filterwarnings("ignore")
BANDS = np.array(range(1050, 2051, 100))          # 11 rating bands we calibrate over
# Use BOTH time controls: each puzzle gives a rapid AND a blitz solve-prob per band.
# Stacking them = two independent observations of the same skill band -> more signal.
RAPID_IDX = [SUCCESS_PROB_COLS.index(f"success_prob_rapid_{b}") for b in BANDS]
BLITZ_IDX = [SUCCESS_PROB_COLS.index(f"success_prob_blitz_{b}") for b in BANDS]


def _posterior(solved: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """Posterior over the 11 rating bands given the solve pattern (uniform prior)."""
    p = np.clip(curves, 1e-3, 1 - 1e-3)            # (k, 11)
    ll = (solved[:, None] * np.log(p) + (1 - solved[:, None]) * np.log(1 - p)).sum(axis=0)
    post = np.exp(ll - ll.max()); post /= post.sum()
    return post


def estimate_rating(solved: np.ndarray, curves: np.ndarray,
                    method: str = "mean") -> tuple[int, np.ndarray]:
    """Estimate user rating from a solve pattern.

    method:
        "mean"   posterior-mean band (smooth, but averages -> central bias)
        "mode"   argmax band         (no averaging -> less central bias)
        "median" posterior-median band
    Returns (estimated_rating, posterior).
    """
    post = _posterior(solved, curves)
    if method == "mode":
        est = int(BANDS[int(np.argmax(post))])
    elif method == "median":
        cdf = np.cumsum(post)
        est = int(BANDS[int(np.searchsorted(cdf, 0.5))])
    else:  # mean
        est = int(np.round((BANDS * post).sum()))
    return est, post


RAPID_COLS = [f"success_prob_rapid_{b}" for b in BANDS]


def _game_phase(themes: str) -> str:
    """Bucket a puzzle's Lichess Themes string into opening/middlegame/endgame/other."""
    if not isinstance(themes, str):
        return "other"
    words = set(themes.split())
    if "opening" in words:
        return "opening"
    if "middlegame" in words:
        return "middlegame"
    if "endgame" in words:
        return "endgame"
    return "other"


# Cap on how much of the calibration pool may come from "opening" puzzles.
# Opening blunder-punishes (e.g. a cheap Qh4+/Qh5+ trap) have unusually steep
# discrimination curves — rating correlates hard with opening knowledge — so a
# pure top-N-by-spread selection lets them crowd out mid/endgame tactics and the
# pool ends up feeling like "spot the opening trap" instead of real puzzles.
# Real dataset base rate for "opening" is ~5.5%; without this cap it was ~62%
# of the saved 200-puzzle pool. Cap set generously above base rate so opening
# puzzles are still represented, just not dominant.
OPENING_PHASE_CAP = 0.20


def build_calibration_pool(
    model, cfg: dict, sample: int | None = None, input_cols: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build calibration pool from ground-truth Lichess solve-rate curves.

    Uses the actual per-rating-band solve probabilities from the dataset CSV
    (not model predictions) — these are real Lichess play data and have genuine
    discrimination power (spread up to 0.88 vs ~0.20 for model predictions).

    Selection is stratified by game phase (opening/middlegame/endgame) so the
    pool isn't dominated by early-opening blunder-punishes, which discriminate
    unusually cleanly but aren't representative "puzzles" — see OPENING_PHASE_CAP.

    Returns:
        curves : (N, 11) ground-truth rapid solve-probability curves
        order  : indices sorted best-discriminating first, opening-capped
        puzzle_meta : list of dicts {puzzle_id, fen, moves_str}
    """
    import pandas as pd
    needed = ["PuzzleId", "FEN", "Moves", "Themes"] + RAPID_COLS
    df = pd.read_csv(cfg["bigdata_cup"]["train_csv"], usecols=needed)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)

    curves = df[RAPID_COLS].values.astype(np.float32)  # (N, 11)
    phase = df["Themes"].map(_game_phase).values

    # filter 1: monotonic — curve should rise with rating
    diffs = np.diff(curves, axis=1)
    downward = (-np.minimum(diffs, 0)).sum(axis=1)
    span = curves.max(axis=1) - curves.min(axis=1)
    reliability = 1.0 - downward / np.maximum(span, 1e-6)
    monotonic_mask = reliability >= 0.8

    # filter 2: spread >= 0.35 so the puzzle actually discriminates
    spread_mask = span >= 0.35

    # filter 3: not trivially easy (low-end solve rate < 0.85) or trivially hard (> 0.60)
    difficulty_mask = (curves[:, 0] < 0.85) & (curves[:, 0] > 0.02)

    valid_mask = monotonic_mask & spread_mask & difficulty_mask

    discrim = span.copy()
    discrim[~valid_mask] = -1

    # Stratified ranking: interleave phase buckets by discrimination rank within
    # each bucket, capping how many "opening" puzzles can land in the first N
    # picks of `order`. We do this by building a priority key that penalizes
    # opening puzzles once they'd exceed OPENING_PHASE_CAP of picks-so-far —
    # simplest robust way is a greedy pass building the order list directly.
    ranked = np.argsort(discrim)[::-1]
    ranked = ranked[discrim[ranked] > -1]  # drop invalid puzzles entirely

    order_list: list[int] = []
    opening_count = 0
    deferred_opening: list[int] = []
    for idx in ranked:
        if phase[idx] == "opening":
            total_so_far = len(order_list) + 1
            if opening_count + 1 > OPENING_PHASE_CAP * total_so_far:
                deferred_opening.append(idx)
                continue
            opening_count += 1
        order_list.append(idx)
    # Fill any remaining slots with deferred opening puzzles (best-discriminating
    # first) so the pool is still fully populated if non-opening puzzles run out.
    order_list.extend(deferred_opening)
    order = np.array(order_list, dtype=int)
    # append invalid puzzles at the end (never selected by save_calibration_pool's
    # top-n, but keeps `order` a full permutation like the previous behavior)
    invalid = np.where(discrim <= -1)[0]
    order = np.concatenate([order, invalid])

    puzzle_meta = [
        {"puzzle_id": row.PuzzleId, "fen": row.FEN, "moves_str": row.Moves}
        for row in df.itertuples()
    ]
    return curves, order, puzzle_meta


def save_calibration_pool(
    pred_rapid: np.ndarray,
    order: np.ndarray,
    puzzle_meta: list[dict],
    path: str,
    n: int = 200,
) -> None:
    """Save the top-n discriminating puzzles to a compact JSON file for fast loading."""
    import json
    top = order[:n].tolist()
    records = [
        {
            "puzzle_id": puzzle_meta[i]["puzzle_id"],
            "fen": puzzle_meta[i]["fen"],
            "moves_str": puzzle_meta[i]["moves_str"],
            "curve": [round(float(v), 4) for v in pred_rapid[i]],
        }
        for i in top
    ]
    with open(path, "w") as f:
        json.dump(records, f)


def load_calibration_pool_from_file(
    path: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load a pre-saved calibration pool. Returns same tuple as build_calibration_pool."""
    import json
    with open(path) as f:
        records = json.load(f)
    pred_rapid = np.array([r["curve"] for r in records], dtype=np.float32)
    order = np.arange(len(records))   # already sorted best-first
    puzzle_meta = [{"puzzle_id": r["puzzle_id"], "fen": r["fen"], "moves_str": r["moves_str"]}
                   for r in records]
    return pred_rapid, order, puzzle_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--n_calib", type=int, default=12, help="number of calibration puzzles")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    bundle = pickle.load(open(Path(cfg["bigdata_cup"]["models_dir"]) / "perplayer_curve_lgbm.pkl", "rb"))
    model = bundle["model"]

    pred_rapid, order, _ = build_calibration_pool(model, cfg, sample=args.sample)

    # Ground-truth blitz curves for the SAME sampled rows as the rapid pool, so the
    # two share one row space and `order` indexes both consistently. (The previous
    # version computed model-predicted blitz over a different holdout slice, which
    # mismatched the full-pool `order` indices and raised IndexError.)
    import pandas as pd
    blitz_cols = [f"success_prob_blitz_{b}" for b in BANDS]
    bdf = pd.read_csv(cfg["bigdata_cup"]["train_csv"],
                      usecols=["PuzzleId"] + RAPID_COLS + blitz_cols)
    if args.sample:
        bdf = bdf.sample(n=min(args.sample, len(bdf)), random_state=42).reset_index(drop=True)
    pred_blitz = bdf[blitz_cols].values.astype(np.float32)   # (N, 11), aligned to pred_rapid

    def make_calib(n_calib: int) -> np.ndarray:
        pick = order[:n_calib]
        return np.vstack([pred_rapid[pick], pred_blitz[pick]])

    band_to_curvecol = {b: i for i, b in enumerate(BANDS)}
    true_ratings = [1050, 1350, 1650, 1950, 2050]

    def mean_abs_err(calib: np.ndarray, method: str, rng) -> float:
        errs = []
        for true_r in true_ratings:
            col = band_to_curvecol[true_r]
            probs = calib[:, col]
            for _ in range(200):
                solved = (rng.random(len(calib)) < probs).astype(float)
                est, _ = estimate_rating(solved, calib, method=method)
                errs.append(abs(est - true_r))
        return float(np.mean(errs))

    print("\n=== improving calibration: estimator x #puzzles (mean abs err, lower=better) ===")
    n_calibs = [12, 15, 30, 60]   # 15 = the deployed frontend session length
    print(f"{'method':>8}" + "".join(f"{f'n={n}':>10}" for n in n_calibs))
    print("-" * (8 + 10 * len(n_calibs)))
    best = (None, 1e9)
    for method in ["mean", "mode", "median"]:
        row = []
        for n in n_calibs:
            rng = np.random.default_rng(cfg["training"]["seed"])
            e = mean_abs_err(make_calib(n), method, rng)
            row.append(e)
            if e < best[1]:
                best = ((method, n), e)
        print(f"{method:>8}" + "".join(f"{e:>10.0f}" for e in row))
    print(f"\nBEST: method={best[0][0]} n={best[0][1]}  ->  mean abs err {best[1]:.0f}")
    print("(baseline rapid-only posterior-mean n=12 was ~256 avg)")

    (bm, bn), _ = best
    calib = make_calib(bn)
    rng = np.random.default_rng(cfg["training"]["seed"])
    print(f"\n=== per-band recovery for BEST config (method={bm}, n={bn}) ===")
    print(f"{'true rating':>12}{'est (mean of 200 sims)':>26}{'mean abs err':>14}")
    print("-" * 52)
    for true_r in true_ratings:
        col = band_to_curvecol[true_r]
        probs = calib[:, col]
        ests, errs = [], []
        for _ in range(200):
            solved = (rng.random(len(calib)) < probs).astype(float)
            est, _ = estimate_rating(solved, calib, method=bm)
            ests.append(est); errs.append(abs(est - true_r))
        print(f"{true_r:>12}{np.mean(ests):>26.0f}{np.mean(errs):>14.0f}")


if __name__ == "__main__":
    main()

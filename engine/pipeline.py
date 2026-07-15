"""
Single-FEN analysis pipeline — the product-facing entry point.

Given a puzzle (FEN + solution moves), produce ALL the ExplainChess outputs in one
call, from a raw position (no dataset lookup, no Maia): the per-player difficulty
curve, a per-band SHAP explanation, a quality/flaw verdict, and — if a user's solve
history is supplied — a calibrated user-rating estimate.

This is PURE PYTHON (no web framework), so the same `analyze_fen()` works in a
notebook, a CLI, or behind a FastAPI route for the full-stack app. The web layer,
when added, only needs to call this function and serialise the returned dict.

Why a single FEN is servable: the per-player model uses ONLY engine + tactical
features (38 of them), both derivable live from the position via Stockfish +
python-chess. It does NOT need the dataset's success_prob columns as input — those
are what the model predicts. So any legal puzzle can be analysed end-to-end.

Cost note: the engine features need live Stockfish analysis (~0.2–1s/FEN at depth 12).
Tactical features and the model/SHAP are sub-100ms. Stockfish is the bottleneck.

Usage (CLI):
    python -m src.pipeline --fen "<FEN>" --moves "e2e4 e7e5 g1f3 ..."

Calibration session (separate flow):
    puzzles = pipe.get_calibration_puzzles(n=60)
    # frontend presents puzzles, checks moves, accumulates outcomes
    result  = pipe.estimate_user_rating([{"curve": p["curve"], "solved": 0|1} for p in ...])
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import chess
import chess.engine
import numpy as np
import shap
import yaml

from engine.explainability.feature_descriptions import describe
from engine.prediction_model.calibration import (
    BANDS, RAPID_IDX, build_calibration_pool, estimate_rating,
    load_calibration_pool_from_file,
)
from engine.quality_pillar.flaw_agent import DECISIVE_CP, UNIQUE_MARGIN, FlawAgent
from engine.stockfish.stockfish_features import (
    _pov_cp,
    decode_eval_columns,
    extract_features_for_puzzle,
)
from engine.tactical.tactical_features import TACTICAL_COLUMNS, extract_tactical_features

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402  (after warnings filter)

def _band_label(p_low: float) -> str:
    """Coarse difficulty label from the weakest-band (1050) solve probability.

    Thresholds are CALIBRATED against real Lichess puzzle ratings, not guessed:
    on a rating-stratified sample of 2,250 dataset puzzles, predicted p_solve_1050
    was compared with the rating-implied difficulty tier (easy < 1200 <= moderate
    < 1600 <= hard < 2000 <= very hard). The original hand-picked cuts
    (0.7/0.4/0.2) sat far above the model's actual output distribution (median
    p1050 for genuinely easy puzzles is ~0.50), mislabelling most easy/moderate
    puzzles as hard — 39% exact tier agreement. The calibrated cuts below give
    ~49% exact and ~89% within-one-tier agreement on a held-back validation half,
    and were chosen from the top candidates for having the best within-one-tier
    score.
    """
    if p_low >= 0.45:
        return "easy"
    if p_low >= 0.28:
        return "moderate"
    if p_low >= 0.15:
        return "hard"
    return "very hard"


class ExplainChessPipeline:
    """Loads the model + opens Stockfish once; analyse many FENs."""

    def __init__(self, config_path: str = "configs/config.yaml", load_calibration: bool = False):
        self.cfg = yaml.safe_load(open(config_path))
        sf = self.cfg["stockfish"]
        stockfish_path = os.environ.get("STOCKFISH_PATH", sf["path"])
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Threads": 1})
        self.engine_name = self.engine.id.get("name", "unknown")
        self.depth = sf["depth"]
        self.depth_candidates = sf["depth_candidates"]
        self.depth_line = sf["depth_line"]
        self.multipv = sf["multipv"]
        self.only_move_margin = sf["only_move_margin"]

        mp = Path(self.cfg["bigdata_cup"]["models_dir"]) / "perplayer_curve_lgbm.pkl"
        bundle = pickle.load(open(mp, "rb"))
        self.model = bundle["model"]
        self.input_cols = bundle["input_cols"]      # 35, the exact training order
        self.target_cols = bundle["target_cols"]    # 22 success_prob names
        # measured per-band holdout MAE = the honest "± confidence" on each prediction
        self.band_mae = bundle.get("band_mae", {})
        self.low_idx = self.target_cols.index("success_prob_rapid_1050")
        self.high_idx = self.target_cols.index("success_prob_rapid_2050")

        # one TreeExplainer per band we explain (weak + strong)
        self._expl_low = shap.TreeExplainer(self.model.estimators_[self.low_idx])
        self._expl_high = shap.TreeExplainer(self.model.estimators_[self.high_idx])

        self.flaw_agent = FlawAgent(stockfish_path)

        # calibration pool — loaded on demand (slow: requires full holdout dataset)
        self._calib_rapid: np.ndarray | None = None   # (N, 11) curves
        self._calib_order: np.ndarray | None = None   # ranked indices
        self._calib_meta: list[dict] | None = None    # puzzle_id, fen, moves_str
        if load_calibration:
            self._load_calibration_pool()

    def close(self):
        self.engine.quit()
        self.flaw_agent.close()

    # ── engine solver: derive a solution line from a bare position ──
    def solve_from_fen(self, fen: str, plies: int = 6, depth: int | None = None) -> dict:
        """Play the engine's principal variation out from `fen` into a solution line.

        For a position the user drew (Board Editor) there is no known solution; the
        engine supplies one. We walk best-move / forced-reply for up to `plies`,
        stopping early at game end or once the line has clearly resolved (a forced
        mate is found, or the eval is no longer changing).

        The returned `moves` uses the "0000" sentinel as moves[0] to signal that
        the FEN is ALREADY the solver's position (no opponent setup move): the
        format is "0000 <solver_move_1> <opp_reply> <solver_move_2> ...", which
        both analyze_fen's feature extractor and the flaw agent understand.
        `solver_is_white` records whose move it is.

        Returns {"fen", "moves", "solver_is_white", "final_eval_cp", "is_mate"}.
        Raises ValueError on an illegal FEN.
        """
        try:
            board = chess.Board(fen)
        except ValueError as e:
            raise ValueError(f"Illegal FEN: {e}") from e
        if board.is_game_over():
            raise ValueError("Position is already terminal — nothing to solve.")

        d = depth if depth is not None else self.depth
        solver = board.turn
        uci: list[str] = []
        last_eval = 0.0
        is_mate = False
        for _ in range(max(1, plies)):
            if board.is_game_over():
                break
            info = self.engine.analyse(board, chess.engine.Limit(depth=d))
            pv = info.get("pv")
            if not pv:
                break
            mv = pv[0]
            last_eval = _pov_cp(info, solver)
            is_mate = info["score"].pov(solver).is_mate()
            uci.append(mv.uci())
            board.push(mv)
            if is_mate:                    # mate found — the combination is decided
                break

        if not uci:
            raise ValueError("Engine produced no moves for this position.")

        return {
            "fen": fen,
            "moves": "0000 " + " ".join(uci),   # 0000 = FEN is already the solver's position
            "solver_is_white": bool(solver == chess.WHITE),
            "final_eval_cp": round(float(last_eval), 0),
            "is_mate": bool(is_mate),
        }

    # ── feature assembly (must match training exactly) ──
    def _feature_vector(self, fen: str, moves_str: str) -> tuple[np.ndarray, dict]:
        raw = extract_features_for_puzzle(
            self.engine, "live", fen, moves_str,
            self.depth, self.depth_candidates, self.multipv,
            self.depth_line, self.only_move_margin)
        eng_df = decode_eval_columns(pd.DataFrame([raw]))
        eng_row = eng_df.iloc[0]
        tac = extract_tactical_features(fen, moves_str)

        merged = {**{c: eng_row.get(c, 0.0) for c in eng_row.index},
                  **{c: tac.get(c, 0.0) for c in TACTICAL_COLUMNS}}
        vec = np.array([merged.get(c, 0.0) for c in self.input_cols], dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0).reshape(1, -1), raw

    # ── input-interpretation guard ──
    BLUNDER_CP = 150.0

    def _solution_loss_cp(self, board: chess.Board, sol_move: chess.Move, game: object) -> float:
        """Eval (cp, solver's POV) lost by playing sol_move, at shallow depth."""
        solver = board.turn
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth_line), game=game)
        before = _pov_cp(info, solver)
        after_board = board.copy()
        after_board.push(sol_move)
        info_after = self.engine.analyse(after_board, chess.engine.Limit(depth=self.depth_line), game=game)
        return before - _pov_cp(info_after, solver)

    def _resolve_interpretation(self, fen: str, moves_str: str) -> tuple[str, dict]:
        """Decide whether moves[0] is really an opponent setup move.

        Returns (moves_str_to_use, input_interpretation dict). When the Lichess
        reading makes the presumed solution a blunder but reading moves[0] as the
        solver's own first move does not, the moves are re-issued with the "0000"
        sentinel (FEN-is-already-solver mode) and the correction is reported.
        """
        tokens = moves_str.strip().split()
        if tokens and tokens[0] == "0000":
            return moves_str, {"mode": "fen_is_solver", "auto_corrected": False, "note": None}
        if len(tokens) < 2:
            return moves_str, {"mode": "lichess", "auto_corrected": False, "note": None}

        board = chess.Board(fen)
        game = object()  # fresh ucinewgame sentinel, same pattern as the extractor

        loss_lichess = None
        try:
            setup = chess.Move.from_uci(tokens[0])
            sol = chess.Move.from_uci(tokens[1])
            b = board.copy()
            if b.is_legal(setup):
                b.push(setup)
                if b.is_legal(sol):
                    loss_lichess = self._solution_loss_cp(b, sol, game)
        except ValueError:
            pass

        loss_as_solver = None
        try:
            first = chess.Move.from_uci(tokens[0])
            if board.is_legal(first):
                loss_as_solver = self._solution_loss_cp(board, first, game)
        except ValueError:
            pass

        lichess_ok = loss_lichess is not None and loss_lichess < self.BLUNDER_CP
        solver_ok = loss_as_solver is not None and loss_as_solver < self.BLUNDER_CP

        if not lichess_ok and solver_ok:
            return "0000 " + moves_str.strip(), {
                "mode": "fen_is_solver", "auto_corrected": True,
                "note": (f"The first move ({tokens[0]}) was treated as the solver's move, not an "
                         f"opponent setup move: under the standard Lichess convention the presumed "
                         f"solution ({tokens[1]}) loses ~{loss_lichess:.0f}cp, so that reading "
                         f"cannot be right."),
            }
        note = None
        if not lichess_ok and not solver_ok and loss_lichess is not None:
            note = (f"Neither reading of the moves gives a sound solution (best interpretation still "
                    f"loses ~{min(x for x in (loss_lichess, loss_as_solver) if x is not None):.0f}cp). "
                    f"Kept the Lichess convention (first move = opponent setup move); results may be unreliable.")
        return moves_str, {"mode": "lichess", "auto_corrected": False, "note": note}

    # ── calibration session ──
    def _load_calibration_pool(self) -> None:
        pool_path = Path(self.cfg["bigdata_cup"]["models_dir"]) / "calibration_pool.json"
        if pool_path.exists():
            self._calib_rapid, self._calib_order, self._calib_meta = load_calibration_pool_from_file(str(pool_path))
        else:
            self._calib_rapid, self._calib_order, self._calib_meta = build_calibration_pool(
                self.model, self.cfg, input_cols=self.input_cols
            )

    def _ensure_calibration(self) -> None:
        if self._calib_rapid is None:
            self._load_calibration_pool()

    def get_calibration_puzzles(self, n: int = 15) -> list[dict]:
        """Return n high-discrimination puzzles for a calibration session.

        Each dict has: puzzle_id, fen, moves_str, curve (list of 11 floats, rapid 1050..2050).
        The frontend uses moves_str.split()[1] to check if the user's move was correct.
        """
        self._ensure_calibration()
        pick = self._calib_order[:n]
        puzzles = []
        for i in pick:
            meta = self._calib_meta[i]
            puzzles.append({
                "puzzle_id": meta["puzzle_id"],
                "fen": meta["fen"],
                "moves_str": meta["moves_str"],
                "curve": [round(float(v), 3) for v in self._calib_rapid[i]],
            })
        return puzzles

    def estimate_user_rating(self, history: list[dict]) -> dict:
        """Estimate a user's rating from their calibration session history.

        history: list of {"curve": [11 floats], "solved": 0 or 1}
        Returns: {"estimated_user_rating": int, "posterior": {band: probability}}
        """
        curves = np.array([h["curve"] for h in history], dtype=np.float32)   # (k, 11)
        solved = np.array([h["solved"] for h in history], dtype=np.float32)  # (k,)
        est, post = estimate_rating(solved, curves, method="mode")
        return {
            "estimated_user_rating": int(est),
            "posterior": {int(b): round(float(post[i]), 3) for i, b in enumerate(BANDS)},
        }

    # ── the all-in-one analysis ──
    def analyze_fen(self, fen: str, moves_str: str) -> dict:
        moves_str, interpretation = self._resolve_interpretation(fen, moves_str)
        x, raw = self._feature_vector(fen, moves_str)
        if not raw.get("extract_ok", False):
            # predicting on the zero-filled fallback vector produces a flat,
            # meaningless curve — surface the failure instead.
            raise ValueError(
                "Feature extraction failed for this puzzle (check that the FEN is legal "
                "and every move in the solution line is legal in sequence)."
            )
        curve = np.clip(self.model.predict(x)[0], 0, 1)        # 22 values
        rapid = curve[RAPID_IDX]                                # 11 (1050..2050)
        p_low, p_high = float(curve[self.low_idx]), float(curve[self.high_idx])

        # 1) difficulty curve — each band carries its MEASURED ± (holdout per-band MAE)
        def band_err(band: int) -> float:
            return round(float(self.band_mae.get(f"success_prob_rapid_{band}", 0.0)), 3)

        rapid_curve = {}
        for i, b in enumerate(BANDS):
            rapid_curve[int(b)] = {"p_solve": round(float(rapid[i]), 3),
                                   "uncertainty_pp": round(band_err(int(b)) * 100, 1)}

        # curve reliability: smoother / more monotonic curve = more trustworthy.
        diffs = np.diff(rapid)
        wrong = -np.minimum(diffs, 0).sum()           # total downward movement (should be ~0)
        raw_span = float(rapid.max() - rapid.min())
        if raw_span < 0.02:
            curve_reliability = None
        else:
            REFERENCE_SPAN = 0.3   # typical 1050->2050 solve-rate range across puzzles
            curve_reliability = round(float(max(0.0, 1.0 - wrong / REFERENCE_SPAN)), 2)

        difficulty = {
            "rapid_curve": rapid_curve,
            "p_solve_1050": round(p_low, 3),
            "p_solve_2050": round(p_high, 3),
            "skill_gap": round(p_high - p_low, 3),
            "label": _band_label(p_low),
            "curve_reliability": curve_reliability,   # 0-1, higher = smoother/monotonic
            "model_mean_uncertainty_pp": round(np.mean(list(self.band_mae.values()) or [0]) * 100, 1),
        }

        # 2) per-band SHAP explanation
        sv_low = self._expl_low.shap_values(x)[0]
        sv_high = self._expl_high.shap_values(x)[0]
        gap_sv = sv_high - sv_low

        def top(sv, k=3):
            return [{"feature": self.input_cols[j], "description": describe(self.input_cols[j]),
                     "shap": round(float(sv[j]), 3),
                     "effect": "easier" if sv[j] > 0 else "harder"}
                    for j in np.argsort(np.abs(sv))[::-1][:k]]

        explanation = {
            "factors_lower_rated_1050": top(sv_low),
            "factors_higher_rated_2050": top(sv_high),
            "skill_gap_drivers": [
                {"feature": self.input_cols[j], "description": describe(self.input_cols[j]),
                 "gap_shap": round(float(gap_sv[j]), 3)}
                for j in np.argsort(np.abs(gap_sv))[::-1][:2]],
        }

        # 3) quality / flaw verdict — with a margin = how decisively it holds
        v = self.flaw_agent.audit("live", fen, moves_str)
        # distance from the nearest decision threshold (bigger = less borderline)
        gap_margin = abs(v.best_gap_cp - UNIQUE_MARGIN)
        adv_margin = abs(abs(v.eval_after_key_cp) - DECISIVE_CP)
        margin = min(gap_margin, adv_margin)
        q_conf = "high" if margin >= 150 else "moderate" if margin >= 50 else "borderline"
        quality = {"verdict": v.verdict, "detail": v.detail,
                   "best_gap_cp": round(v.best_gap_cp, 0),
                   "eval_after_key_cp": round(v.eval_after_key_cp, 0),
                   "confidence": q_conf,
                   "margin_cp": round(float(margin), 0)}

        # 4) diagnostics — make silent-garbage cases visible to the caller
        swing = raw.get("eval_swing")
        loss_cp = round(float(max(0.0, -swing)), 0) if swing is not None and np.isfinite(swing) else None
        warning = None
        if loss_cp is not None and loss_cp >= self.BLUNDER_CP:
            warning = (f"The provided solution move loses ~{loss_cp:.0f}cp against best play, so it "
                       f"is not close to engine-best. The model is trained on puzzles whose solution "
                       f"IS the engine-best move; trap-style or unsound puzzles will be misjudged and "
                       f"this difficulty estimate is unreliable.")
        diagnostics = {
            "engine": self.engine_name,
            "extract_ok": bool(raw.get("extract_ok", False)),
            "solution_eval_loss_cp": loss_cp,
            "warning": warning,
        }

        # assemble in display order: prediction & verdict first, explanation LAST
        result = {"difficulty": difficulty, "quality": quality, "explanation": explanation,
                  "input_interpretation": interpretation, "diagnostics": diagnostics}
        return result


def _format_summary(out: dict) -> str:
    """Human-readable summary of the analysis (printed after the JSON for the CLI)."""
    d, q = out["difficulty"], out["quality"]
    lines = ["", "=" * 60, "  EXPLAINCHESS — PUZZLE ANALYSIS", "=" * 60]

    # difficulty + a small ASCII curve, with measured ± per band
    lines.append(f"\n  DIFFICULTY: {d['label'].upper()}   "
                 f"(solve rate 1050 {d['p_solve_1050']*100:.0f}%  ->  "
                 f"2050 {d['p_solve_2050']*100:.0f}%,  skill gap {d['skill_gap']*100:.0f}pp)")
    rel = d['curve_reliability']
    rel_str = "n/a (curve too flat to grade)" if rel is None else f"{rel:.2f}/1.00"
    lines.append(f"  curve reliability {rel_str}   "
                 f"(model accuracy: ±{d['model_mean_uncertainty_pp']:.0f}pp mean per band)")
    lines.append("  solve-probability curve by rating band (± measured uncertainty):")
    for b, cell in d["rapid_curve"].items():
        p, u = cell["p_solve"], cell["uncertainty_pp"]
        bar = "#" * int(round(p * 40))
        lines.append(f"    {b}  {p*100:5.0f}% (±{u:.0f}pp)  {bar}")

    # quality verdict + how decisively it holds
    mark = "OK" if q["verdict"] == "OK" else "FLAGGED"
    lines.append(f"\n  QUALITY: {q['verdict']} [{mark}] — {q['detail']}")
    lines.append(f"    confidence: {q['confidence']} (margin {q['margin_cp']:.0f}cp from threshold)")

    # explanation, last
    lines.append("\n  EXPLANATION — key contributing factors (SHAP):")
    lines.append("    for lower-rated (1050) players:")
    for f in out["explanation"]["factors_lower_rated_1050"]:
        lines.append(f"      • {f['description']:<42} ({f['effect']})")
    lines.append("    factors that most affect the skill gap:")
    for f in out["explanation"]["skill_gap_drivers"]:
        lines.append(f"      • {f['description']}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    ap.add_argument("--moves", required=True, help="space-separated UCI moves (opponent move first)")
    ap.add_argument("--json", action="store_true", help="print raw JSON only (default also prints a readable summary)")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    pipe = ExplainChessPipeline(args.config)
    try:
        out = pipe.analyze_fen(args.fen, args.moves)
        print(json.dumps(out, indent=2))
        if not args.json:
            print(_format_summary(out))
    finally:
        pipe.close()


if __name__ == "__main__":
    main()

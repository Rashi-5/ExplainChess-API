"""
Human-readable descriptions for every feature in the cup fusion / servable models.

Used by the per-puzzle explanation layer to turn raw feature names into plain
English. Covers all four groups: engine (Stockfish, decoded), success_prob (Maia),
curve, and tactical. Also provides a feature->group mapper and a directional
phrasing helper (does a higher value make the puzzle harder or easier?).
"""

from __future__ import annotations

from engine.stockfish.cup_data import CURVE_COLS, SUCCESS_PROB_COLS
from engine.stockfish.dataset import FEATURE_COLUMNS
from engine.tactical.tactical_features import TACTICAL_COLUMNS

# ── plain-English descriptions ────────────────────────────────────────────────

FEATURE_DESCRIPTIONS: dict[str, str] = {
    # engine (decoded eval channels)
    "cp_eval_cp": "engine evaluation of the position (clipped centipawns)",
    "cp_eval_mate_ply": "forced-mate distance at the position",
    "cp_eval_d1_cp": "shallow (depth-1) engine evaluation",
    "cp_eval_d1_mate_ply": "shallow forced-mate distance",
    "cp_eval_after_cp": "engine evaluation after the solution move",
    "cp_eval_after_mate_ply": "forced-mate distance after the solution move",
    "eval_delta": "shift in evaluation from shallow to full-depth search",
    "eval_swing": "evaluation swing caused by the solution move",
    "best_move_depth": "search depth needed to find the best move",
    "num_candidate_moves": "number of near-equal candidate moves (100cp)",
    "num_candidate_moves_50cp": "number of near-equal candidate moves (50cp)",
    "num_solution_moves": "length of the solution line",
    "material_balance": "material balance",
    "total_material": "total material on the board",
    "num_pieces": "number of pieces on the board",
    "king_safety_white": "white king exposure",
    "king_safety_black": "black king exposure",
    "is_check": "the position involves a check",
    "is_capture": "the solution starts with a capture",
    "is_promotion": "the solution involves a promotion",
    "is_mate": "the position leads to forced mate",
    "pawn_structure_score": "pawn-structure weakness",
    "mobility_diff": "mobility difference between sides",
    "first_move_is_sacrifice": "the first solution move sacrifices material",
    # tactical
    "num_legal_moves": "number of legal moves available",
    "num_checking_moves": "number of checking moves available",
    "num_captures_available": "number of captures available",
    "num_undefended_own": "solver's undefended pieces",
    "num_undefended_opp": "opponent's undefended pieces",
    "mate_in_1_available": "a mate-in-one is available",
    "is_recapture": "the solution is a recapture",
    "first_move_gives_check": "the solution's first move gives check",
    "first_move_is_capture": "the solution's first move is a capture",
    "material_imbalance_abs": "absolute material imbalance",
    "side_to_move": "side to move (white/black)",
    "in_check_at_start": "the solver is in check at the start",
    "num_attacked_own": "solver's pieces currently under attack",
    "can_interpose_check": "a check can be blocked by interposition",
}

# success_prob and curve: generated programmatically (22 + 10 names)
for _c in SUCCESS_PROB_COLS:
    _tc, _r = _c.replace("success_prob_", "").split("_")
    FEATURE_DESCRIPTIONS[_c] = f"human solve rate for {_r}-rated players ({_tc})"
for _c in CURVE_COLS:
    FEATURE_DESCRIPTIONS[_c] = f"summary of the human solve-rate curve ({_c.replace('_prob_', ' ')})"


def describe(feature: str) -> str:
    """Human-readable description, falling back to a cleaned-up name."""
    return FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " "))


def group_of(feature: str) -> str:
    """Map a feature to its group: engine / success_prob / curve / tactical."""
    if feature in SUCCESS_PROB_COLS:
        return "success_prob"
    if feature in CURVE_COLS:
        return "curve"
    if feature in TACTICAL_COLUMNS:
        return "tactical"
    return "engine"


# Human-friendly group labels for explanations.
GROUP_LABELS: dict[str, str] = {
    "engine": "engine analysis",
    "success_prob": "human solve statistics",
    "curve": "human solve-rate curve",
    "tactical": "tactical position features",
}

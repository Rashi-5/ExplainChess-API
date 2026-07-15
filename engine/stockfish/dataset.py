"""
FEATURE_COLUMNS — the engine feature names used by the per-player model.

(The old torch Dataset/DataLoader/PuzzleDataset scaffolding for the legacy
CNN/MLP pipeline lived here and has been removed; it's not part of this repo
per the README, and nothing imports it — only FEATURE_COLUMNS is used.)
"""

from __future__ import annotations

# 24 features. The three raw eval columns (cp_eval, cp_eval_d1, cp_eval_after)
# are NOT used directly — they pack mate-in-N into a sentinel band that corrupts
# the continuous scale. decode_eval_columns() splits each into a clipped *_cp
# channel + a signed *_mate_ply channel; eval_delta/eval_swing are recomputed on
# the clipped parts. Run decode_eval_columns(df) before selecting these columns.
FEATURE_COLUMNS: list[str] = [
    "cp_eval_cp",
    "cp_eval_mate_ply",
    "cp_eval_d1_cp",
    "cp_eval_d1_mate_ply",
    "cp_eval_after_cp",
    "cp_eval_after_mate_ply",
    "eval_delta",
    "eval_swing",
    "best_move_depth",
    "num_candidate_moves",
    "num_candidate_moves_50cp",
    "num_solution_moves",
    "material_balance",
    "total_material",
    "num_pieces",
    "king_safety_white",
    "king_safety_black",
    "is_check",
    "is_capture",
    "is_promotion",
    "is_mate",
    "pawn_structure_score",
    "mobility_diff",
    "first_move_is_sacrifice",
]

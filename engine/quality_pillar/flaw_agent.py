"""
Agentic puzzle-quality auditor — flags quality DEFECTS in chess puzzles.

This is an AGENT (not an LLM): an autonomous perceive -> probe -> decide loop that
uses Stockfish as a tool to audit a puzzle and reach a verdict. It re-probes the
engine LIVE, so it works on ANY FEN, not just dataset rows (servable). It is the
computational analogue of Lichess's own puzzle reporter.

Motivation: Lichess pre-filters puzzles for uniqueness, so in the corpus ~98.8% are
clean (one solution within 50cp). The useful, objective task is therefore FLAW
DETECTION — surfacing the rare defective puzzles. The verdicts are engine-verifiable,
not subjective (unlike human Popularity, which we showed is orthogonal to engine
quality — see src/quality/quality_study.py).

Lichess puzzle convention: the FEN is BEFORE the opponent's setup move; Moves[0] is
the opponent move, then the SOLVER replies with Moves[1], Moves[3], ... The audit is
done at the position the solver actually faces (after the opponent's first move), from
the SOLVER's point of view.

Tools the agent uses:
  - probe_multipv(board): top-k engine lines -> uniqueness gap
  - probe_best_move(board): is the claimed key move the engine's best?

Verdicts:
  OK         unique best move AND a decisive advantage
  AMBIGUOUS  2+ moves within UNIQUE_MARGIN cp of best  (multiple solutions)
  WRONG_KEY  the claimed solution move is not the engine's best move
  WEAK       advantage after the key move < DECISIVE_CP  (no real win to find)

Usage:
    python -m src.quality.flaw_agent --sample 200            # audit dataset puzzles
    python -m src.quality.flaw_agent --fen "<FEN>" --moves "e2e4 e7e5 ..."   # single
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import chess
import chess.engine
import pandas as pd
import yaml

from engine.stockfish.stockfish_features import _pov_cp

warnings.filterwarnings("ignore")

# thresholds (objective, engine-based; documented so they're defensible)
UNIQUE_MARGIN = 50      # cp: a 2nd move within this of best => ambiguous (multiple solutions)
DECISIVE_CP = 150       # cp: solver should reach at least this advantage, else "weak"
WRONG_KEY_MARGIN = 200  # cp: only flag WRONG_KEY if the stated key is THIS much worse than
                        # best. Raised from 100 to avoid false flags when two moves are both
                        # clearly winning but Stockfish slightly prefers one over the other.
DEPTH = 14              # audit search depth; 20 was too slow for live serving, 12 too shallow
                        # to trust uniqueness/decisiveness verdicts. 14 is the balance.
MULTIPV = 5             # raised from 3: key move ranked 4th/5th no longer wrongly flagged
MATE_DECISIVE_CP = 10000  # a forced mate counts as decisive regardless of the cp threshold


@dataclass
class Verdict:
    puzzle_id: str
    verdict: str
    best_gap_cp: float       # eval(best) - eval(2nd best), solver POV
    key_is_best: bool        # claimed solution == engine best?
    eval_after_key_cp: float # advantage after the solver plays the key move
    detail: str


class FlawAgent:
    """Stockfish-driven agent that audits one puzzle to a quality verdict."""

    def __init__(self, engine_path: str):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine.configure({"Threads": 1})

    def close(self):
        self.engine.quit()

    # ── tools ──
    def _probe_multipv(self, board: chess.Board, pov: chess.Color):
        info = self.engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=MULTIPV)
        if not isinstance(info, list):
            info = [info]
        lines = [(i["pv"][0], _pov_cp(i, pov)) for i in info if i.get("pv")]
        return lines  # [(move, cp_from_solver_pov)], sorted best-first by engine

    @staticmethod
    def _is_mate_score(cp: float) -> bool:
        # _pov_cp maps mates to |cp| >= 5000 (mate-in-50 -> 5000, mate-in-1 -> 9900).
        return abs(cp) >= 5000.0

    def _validate_line(self, board: chess.Board, solver: chess.Color, moves: list[str]) -> str | None:
        """Walk the FULL solution line from moves[1] onward, checking every SOLVER
        move is still the engine's best (or within UNIQUE_MARGIN). Returns a defect
        detail string if the continuation breaks down, else None.

        Lichess convention: moves alternate solver, opponent, solver, ... starting at
        moves[1]. Opponent replies (the even indices after the setup move) are assumed
        forced and just played onto the board; only the solver's moves are audited.
        """
        b = board.copy()
        for idx in range(1, len(moves)):
            mv = moves[idx]
            if mv == "0000":
                break
            is_solver_move = (b.turn == solver)
            if is_solver_move:
                lines = self._probe_multipv(b, solver)
                if not lines:
                    return f"no engine lines at solver move {idx}"
                best_mv, best_cp = lines[0]
                second_cp = lines[1][1] if len(lines) > 1 else best_cp - 10000
                if best_mv.uci() != mv:
                    # find the played move's eval among the lines
                    played_cp = next((cp for m, cp in lines if m.uci() == mv), None)
                    if played_cp is None or (best_cp - played_cp) >= WRONG_KEY_MARGIN:
                        return (f"continuation move {idx} ({mv}) is not best "
                                f"(engine prefers {best_mv.uci()})")
                # a non-decisive, non-forcing solver move mid-line means the puzzle peters out
                if not self._is_mate_score(best_cp) and abs(best_cp) < DECISIVE_CP and best_cp - second_cp < UNIQUE_MARGIN:
                    return f"continuation at move {idx} is neither decisive nor unique"
            try:
                b.push(chess.Move.from_uci(mv))
            except Exception:
                return f"move {idx} ({mv}) is illegal in the line"
        return None

    # ── the agent loop ──
    def audit(self, puzzle_id: str, fen: str, moves_str: str) -> Verdict:
        board = chess.Board(fen)
        moves = moves_str.split()

        if len(moves) >= 2 and moves[0] != "0000":
            # Lichess convention: play the opponent's setup move first
            board.push(chess.Move.from_uci(moves[0]))
            solver = board.turn
            key_move = moves[1]
        elif len(moves) >= 2 and moves[0] == "0000":
            # FEN-only mode: FEN is already the solver's position
            solver = board.turn
            key_move = moves[1]
        else:
            return Verdict(puzzle_id, "FLAWED", 0, False, 0, "no moves")

        lines = self._probe_multipv(board, solver)
        if not lines:
            return Verdict(puzzle_id, "FLAWED", 0, False, 0, "no engine lines")

        best_move, best_cp = lines[0]
        second_cp = lines[1][1] if len(lines) > 1 else best_cp - 10000
        gap = best_cp - second_cp

        key_is_best = (key_move is not None and best_move.uci() == key_move)

        # advantage the solver reaches after the key move
        eval_after = best_cp  # engine's best == what a correct solve achieves
        key_in_lines = key_is_best
        if key_move and not key_is_best:
            for mv, cp in lines:
                if mv.uci() == key_move:
                    eval_after = cp
                    key_in_lines = True
                    break
            if not key_in_lines:
                # key move not in top-MULTIPV lines — evaluate it directly instead of
                # assuming it's terrible (avoids false WRONG_KEY on valid but less common moves)
                try:
                    key_board = board.copy()
                    key_board.push(chess.Move.from_uci(key_move))
                    key_info = self.engine.analyse(key_board, chess.engine.Limit(depth=DEPTH))
                    # negate because we're now from the opponent's POV
                    eval_after = -_pov_cp(key_info, key_board.turn)
                    key_in_lines = True
                except Exception:
                    eval_after = best_cp - 10000
        key_deficit = best_cp - eval_after             # how much worse the key is vs best

        # WRONG_KEY only if the stated key is MEANINGFULLY worse than best — a different
        # but comparable winning move (e.g. two recaptures) is not a defect.
        is_wrong_key = (key_move is not None and not key_is_best
                        and key_deficit >= WRONG_KEY_MARGIN)

        # a forced mate is decisive regardless of the cp threshold (mate-awareness):
        # _pov_cp encodes mates as |cp| >= 5000, which already clears DECISIVE_CP, but
        # we make the intent explicit so the WEAK rule can never fire on a mating line.
        key_is_mate = self._is_mate_score(eval_after)
        is_decisive = key_is_mate or abs(eval_after) >= DECISIVE_CP

        # ── decide ──
        if is_wrong_key:
            v, detail = "WRONG_KEY", (f"claimed {key_move} is {key_deficit:.0f}cp worse than "
                                      f"engine best {best_move.uci()}")
        elif gap < UNIQUE_MARGIN:
            v, detail = "AMBIGUOUS", f"2nd move within {gap:.0f}cp of best (multiple solutions)"
        elif not is_decisive:
            v, detail = "WEAK", f"advantage only {eval_after:.0f}cp (no decisive win)"
        else:
            # first move is clean — now verify the REST of the solution line holds up.
            line_defect = self._validate_line(board, solver, moves)
            if line_defect:
                v, detail = "WEAK", f"solution breaks down: {line_defect}"
            else:
                adv = "forced mate" if key_is_mate else f"adv {eval_after:.0f}cp"
                v, detail = "OK", f"unique best move, gap {gap:.0f}cp, {adv}"

        return Verdict(puzzle_id, v, gap, key_is_best, eval_after, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--fen", default=None)
    ap.add_argument("--moves", default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    agent = FlawAgent(cfg["stockfish"]["path"])

    try:
        if args.fen and args.moves:
            v = agent.audit("single", args.fen, args.moves)
            print(f"VERDICT: {v.verdict}\n  {v.detail}")
            return

        cup = cfg["bigdata_cup"]
        df = pd.read_csv(cup["train_csv"], usecols=["PuzzleId", "FEN", "Moves"]).head(args.sample)
        counts: dict[str, int] = {}
        flagged = []
        for r in df.itertuples():
            v = agent.audit(r.PuzzleId, r.FEN, r.Moves)
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
            if v.verdict != "OK":
                flagged.append(v)

        print(f"\n=== AGENTIC FLAW AUDIT — {len(df)} puzzles (depth {DEPTH}, multipv {MULTIPV}) ===")
        for k in ["OK", "AMBIGUOUS", "WEAK", "WRONG_KEY", "FLAWED"]:
            if k in counts:
                print(f"  {k:11} {counts[k]:>5} ({100*counts[k]/len(df):.1f}%)")
        print(f"\nflagged {len(flagged)} puzzles. examples:")
        for v in flagged[:12]:
            print(f"  [{v.verdict:10}] {v.puzzle_id}  {v.detail}")
        print("\nNOTE: flaws are rare (Lichess pre-filters), so report precision/recall on the")
        print("flagged class + manual inspection, not accuracy. Verdicts are engine-objective.")
    finally:
        agent.close()


if __name__ == "__main__":
    main()

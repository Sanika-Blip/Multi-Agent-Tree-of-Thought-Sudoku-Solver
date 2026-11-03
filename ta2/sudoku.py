"""
sudoku.py
---------
Core representation + rule-checking for the 4x4 mini-Sudoku used in this
Tree-of-Thought vs Chain-of-Thought demo.

Board layout (rows/cols 0-3, values 1-4, box size 2x2):

        c0 c1 | c2 c3
      +-------+-------+
   r0 |  .  . |  .  . |
   r1 |  .  . |  .  . |
      +-------+-------+
   r2 |  .  . |  .  . |
   r3 |  .  . |  .  . |
      +-------+-------+

Rule: every row, every column, and each of the four 2x2 boxes must contain
1,2,3,4 exactly once.
"""

from copy import deepcopy
from typing import List, Optional, Tuple

N = 4
BOX = 2
EMPTY = 0

# The puzzle used for this jury demo. Hand-picked to have exactly ONE
# solution and to be hard enough that a naive single-shot guess is likely
# to hit at least one dead end -- which is exactly the point of the demo.
PUZZLE = [
    [2, 0, 0, 0],
    [0, 0, 2, 3],
    [0, 0, 0, 0],
    [1, 0, 0, 4],
]


def clone(grid: List[List[int]]) -> List[List[int]]:
    return deepcopy(grid)


def box_id(r: int, c: int) -> int:
    return (r // BOX) * BOX + (c // BOX)


def is_valid_placement(grid: List[List[int]], r: int, c: int, val: int) -> Tuple[bool, str]:
    """Check whether placing `val` at (r, c) violates row/col/box rules.
    Returns (is_valid, reason_if_invalid)."""
    for cc in range(N):
        if cc != c and grid[r][cc] == val:
            return False, f"row {r} already has {val} at column {cc}"
    for rr in range(N):
        if rr != r and grid[rr][c] == val:
            return False, f"column {c} already has {val} at row {rr}"
    br, bc = (r // BOX) * BOX, (c // BOX) * BOX
    for rr in range(br, br + BOX):
        for cc in range(bc, bc + BOX):
            if (rr, cc) != (r, c) and grid[rr][cc] == val:
                return False, f"box {box_id(r, c)} already has {val} at ({rr},{cc})"
    return True, ""


def find_empty(grid: List[List[int]], strategy: str = "first") -> Optional[Tuple[int, int]]:
    """Pick the next empty cell to work on.

    strategy="first" : plain left-to-right, top-to-bottom (naive order).
        Used by the ToT demo on purpose -- it's the order a person would
        follow by hand, and it's exactly the order that occasionally
        commits to a locally-valid but globally-wrong candidate, which is
        what makes the resulting dead-ends/backtracks worth documenting.
    strategy="mrv"   : Minimum-Remaining-Values heuristic (fewest legal
        candidates first) -- the standard CSP speed-up. Left in for
        anyone who wants to show the jury how much smaller the tree gets
        with a smarter cell-selection strategy.
    """
    if strategy == "mrv":
        best = None
        best_count = 5
        for r in range(N):
            for c in range(N):
                if grid[r][c] == EMPTY:
                    candidates = [v for v in range(1, 5) if is_valid_placement(grid, r, c, v)[0]]
                    if len(candidates) < best_count:
                        best = (r, c)
                        best_count = len(candidates)
        return best

    for r in range(N):
        for c in range(N):
            if grid[r][c] == EMPTY:
                return (r, c)
    return None


def is_solved(grid: List[List[int]]) -> bool:
    return all(grid[r][c] != EMPTY for r in range(N) for c in range(N))


def grid_to_str(grid: List[List[int]]) -> str:
    lines = []
    for r in range(N):
        row = []
        for c in range(N):
            row.append(str(grid[r][c]) if grid[r][c] != EMPTY else ".")
            if c == 1:
                row.append("|")
        lines.append(" ".join(row))
        if r == 1:
            lines.append("-" * 11)
    return "\n".join(lines)


def full_grid_violations(grid: List[List[int]]) -> List[str]:
    """Used to grade a *complete* grid a model claims is the answer
    (used by the Chain-of-Thought solver, which fills everything at once)."""
    problems = []
    for r in range(N):
        vals = [grid[r][c] for c in range(N)]
        if sorted(vals) != [1, 2, 3, 4]:
            problems.append(f"row {r} is {vals}, not a permutation of 1-4")
    for c in range(N):
        vals = [grid[r][c] for r in range(N)]
        if sorted(vals) != [1, 2, 3, 4]:
            problems.append(f"column {c} is {vals}, not a permutation of 1-4")
    for br in range(0, N, BOX):
        for bc in range(0, N, BOX):
            vals = [grid[r][c] for r in range(br, br + BOX) for c in range(bc, bc + BOX)]
            if sorted(vals) != [1, 2, 3, 4]:
                problems.append(f"box at ({br},{bc}) is {vals}, not a permutation of 1-4")
    for r in range(N):
        for c in range(N):
            if grid[r][c] == EMPTY:
                problems.append(f"cell ({r},{c}) was left empty")
    return problems


def brute_force_solutions(grid: List[List[int]], limit: int = 2) -> List[List[List[int]]]:
    """Ground-truth solver (plain backtracking, no LLM) used ONLY to verify
    the puzzle has a unique solution before we hand it to the models."""
    solutions: List[List[List[int]]] = []

    def solve(g):
        if len(solutions) >= limit:
            return
        pos = find_empty(g)
        if pos is None:
            solutions.append(clone(g))
            return
        r, c = pos
        for v in range(1, 5):
            ok, _ = is_valid_placement(g, r, c, v)
            if ok:
                g[r][c] = v
                solve(g)
                g[r][c] = EMPTY
                if len(solutions) >= limit:
                    return

    solve(clone(grid))
    return solutions


if __name__ == "__main__":
    print("Puzzle:\n" + grid_to_str(PUZZLE))
    sols = brute_force_solutions(PUZZLE, limit=2)
    print(f"\nSolutions found (capped at 2): {len(sols)}")
    for s in sols:
        print(grid_to_str(s))
        print()


# ---------------------------------------------------------------------------
# TA2 additions: the 5 evaluation puzzles (same set is run on TA1 and TA2)
# and a compact board key used by the long-term dead-end memory.
# Every puzzle below was checked with brute_force_solutions() to have
# exactly one solution.
# ---------------------------------------------------------------------------

PUZZLES = {
    "P1": PUZZLE,  # the TA1 jury puzzle
    "P2": [[0, 2, 0, 3], [3, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0]],
    "P3": [[0, 0, 0, 4], [3, 0, 0, 0], [4, 0, 0, 1], [1, 0, 0, 0]],
    "P4": [[1, 4, 2, 0], [0, 0, 4, 0], [0, 2, 0, 0], [4, 0, 0, 0]],
    "P5": [[0, 0, 3, 1], [0, 0, 0, 0], [4, 1, 2, 0], [0, 0, 0, 4]],
}


def board_key(grid: List[List[int]]) -> str:
    """16-char string, row-major, '0' for empty. Used as a memory key."""
    return "".join(str(v) for row in grid for v in row)

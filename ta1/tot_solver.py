"""
tot_solver.py
-------------
Step 3 of the assignment: solve the same puzzle with a Tree-of-Thought
approach. At each cell, generate 2-3 candidate placements, evaluate each
for rule violations, and backtrack from dead ends.

Step 4: every node explored (kept AND pruned) is logged into a SearchNode
tree so the full tree -- including pruned branches -- can be documented
and diagrammed (see tree_viz.py).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import itertools

from sudoku import (
    PUZZLE, N, clone, is_valid_placement, find_empty, is_solved, grid_to_str,
)


_id_counter = itertools.count(1)


@dataclass
class SearchNode:
    id: int
    parent_id: Optional[int]
    cell: Optional[tuple]          # (r, c) this node placed a value into
    value: Optional[int]           # value tried at that cell
    status: str                    # "root" | "valid" | "pruned" | "solution" | "backtracked"
    reason: str                    # why pruned, or why chosen
    depth: int
    children: List["SearchNode"] = field(default_factory=list)


@dataclass
class ToTResult:
    root: SearchNode
    solved_grid: Optional[List[List[int]]]
    nodes_explored: int
    nodes_pruned: int
    backtracks: int
    backend_name: str
    log_lines: List[str] = field(default_factory=list)


def run_tot(backend, puzzle: List[List[int]] = None, max_candidates: int = 3) -> ToTResult:
    puzzle = puzzle or PUZZLE
    root = SearchNode(id=0, parent_id=None, cell=None, value=None,
                       status="root", reason="initial board", depth=0)

    stats = {"explored": 0, "pruned": 0, "backtracks": 0}
    log: List[str] = []
    backend_name = ["mock"]

    def search(grid, parent_node: SearchNode, depth: int) -> Optional[List[List[int]]]:
        if is_solved(grid):
            return grid

        pos = find_empty(grid, strategy="first")
        if pos is None:
            return None
        r, c = pos

        candidates = backend.propose_candidates(grid, r, c, k=max_candidates)
        log.append(f"[depth {depth}] Impasse at cell ({r},{c}). "
                    f"Model proposes candidates: {candidates}")

        # Safety-net fallback: the model is only asked for k candidates and
        # can genuinely omit the one digit that turns out to be necessary
        # (observed in practice with small local models). Rather than let
        # the whole search fail on that, append any remaining logically
        # valid digits the model didn't suggest, clearly flagged as
        # fallback so the log is honest about which candidates came from
        # the model vs. the validator's own exhaustive check.
        all_valid = [v for v in range(1, 5) if is_valid_placement(grid, r, c, v)[0]]
        fallback_extra = [v for v in all_valid if v not in candidates]
        if fallback_extra:
            log.append(f"  [fallback] model's candidates missed valid digit(s) "
                        f"{fallback_extra} for ({r},{c}) -- adding them so the "
                        f"search doesn't dead-end on a model omission.")
        ordered_candidates = list(candidates) + fallback_extra

        for val in ordered_candidates:
            ok, reason = is_valid_placement(grid, r, c, val)
            node_id = next(_id_counter)
            stats["explored"] += 1
            is_fallback = val in fallback_extra

            if not ok:
                stats["pruned"] += 1
                node = SearchNode(
                    id=node_id, parent_id=parent_node.id, cell=(r, c), value=val,
                    status="pruned", reason=f"invalid: {reason}", depth=depth + 1,
                )
                parent_node.children.append(node)
                log.append(f"  -> try {val} at ({r},{c}): INVALID ({reason}). Pruned.")
                continue

            tag = " [fallback candidate, not suggested by model]" if is_fallback else ""
            node = SearchNode(
                id=node_id, parent_id=parent_node.id, cell=(r, c), value=val,
                status="valid",
                reason="passes row/col/box check" + (" (fallback)" if is_fallback else ""),
                depth=depth + 1,
            )
            parent_node.children.append(node)
            log.append(f"  -> try {val} at ({r},{c}): valid, descending.{tag}")

            next_grid = clone(grid)
            next_grid[r][c] = val
            result = search(next_grid, node, depth + 1)
            if result is not None:
                node.status = "solution"
                return result

            # dead end further down the tree -- backtrack
            stats["backtracks"] += 1
            node.status = "backtracked"
            node.reason += " (later impasse had no valid candidates -- backtracked)"
            log.append(f"  <- backtracking from ({r},{c})={val}: "
                        f"no candidate below it led to a full solution.")

        return None

    solved = search(clone(puzzle), root, 0)

    if hasattr(backend, "generate_cot"):
        backend_name[0] = "mock:qwen2.5:3b(offline-sim)"
    else:
        backend_name[0] = getattr(backend, "model", "ollama")

    return ToTResult(
        root=root,
        solved_grid=solved,
        nodes_explored=stats["explored"],
        nodes_pruned=stats["pruned"],
        backtracks=stats["backtracks"],
        backend_name=backend_name[0],
        log_lines=log,
    )


def print_tree(node: SearchNode, indent: int = 0):
    label = "ROOT" if node.cell is None else f"({node.cell[0]},{node.cell[1]})={node.value}"
    print("  " * indent + f"[{node.status}] {label}  -- {node.reason}")
    for child in node.children:
        print_tree(child, indent + 1)


if __name__ == "__main__":
    from llm_backend import MockBackend

    result = run_tot(MockBackend())
    print("=== Search log ===")
    for line in result.log_lines:
        print(line)
    print("\n=== Search tree ===")
    print_tree(result.root)
    print(f"\nNodes explored: {result.nodes_explored}  Pruned: {result.nodes_pruned}  "
          f"Backtracks: {result.backtracks}")
    print("\n=== Solved grid ===")
    print(grid_to_str(result.solved_grid) if result.solved_grid else "NO SOLUTION FOUND")
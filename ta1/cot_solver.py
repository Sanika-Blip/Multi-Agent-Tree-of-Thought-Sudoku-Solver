"""
cot_solver.py
-------------
Step 2 of the assignment: attempt the puzzle with a single Chain-of-Thought
prompt and note where it goes wrong (if it does).

This sends ONE prompt asking the model to reason step-by-step and produce a
complete filled grid. There is no tool-calling, no validity checking loop,
no backtracking -- if the model paints itself into a corner, it has no way
to recover, which is exactly the limitation this demo is meant to expose.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from sudoku import PUZZLE, N, grid_to_str, full_grid_violations, clone


COT_PROMPT_TEMPLATE = """You are solving a 4x4 mini-Sudoku. Digits 1-4 only.
Rules: each row, each column, and each 2x2 box must contain 1,2,3,4 exactly once.
'.' marks an empty cell that you must fill in.

Puzzle:
{puzzle}

Think through the puzzle step by step, one cell at a time, left to right,
top to bottom, explaining which digit you place in each empty cell and why.
Do NOT skip cells. At the end, output the completed grid on its own lines,
prefixed exactly with the line 'FINAL GRID:' followed by 4 lines of 4
space-separated digits.
"""


@dataclass
class CoTResult:
    prompt: str
    transcript: str
    backend_name: str
    parsed_grid: Optional[List[List[int]]]
    violations: List[str] = field(default_factory=list)
    correct: bool = False


def format_puzzle(grid: List[List[int]]) -> str:
    return "\n".join(
        " ".join(str(v) if v != 0 else "." for v in row) for row in grid
    )


def parse_final_grid(text: str) -> Optional[List[List[int]]]:
    """Pull the 4x4 grid out of the model's free-form response."""
    if "FINAL GRID:" in text:
        tail = text.split("FINAL GRID:", 1)[1]
    else:
        tail = text
    # find every line that looks like 4 digits (space or no space separated)
    rows = []
    for line in tail.splitlines():
        digits = re.findall(r"[1-4]", line)
        if len(digits) == 4:
            rows.append([int(d) for d in digits])
        if len(rows) == 4:
            break
    if len(rows) == 4:
        return rows
    return None


def run_cot(backend, puzzle: List[List[int]] = None) -> CoTResult:
    puzzle = puzzle or PUZZLE
    prompt = COT_PROMPT_TEMPLATE.format(puzzle=format_puzzle(puzzle))

    if hasattr(backend, "generate_cot"):
        # MockBackend: deterministic offline simulation
        resp = backend.generate_cot(puzzle)
    else:
        # Real OllamaBackend -- first call can be slow (model load into
        # memory), especially on CPU-only machines. Print progress so this
        # doesn't look hung.
        print("[CoT] Sending single-shot prompt to Ollama... "
              "(first response can take 30s-a few minutes on CPU)")
        resp = backend.generate(prompt)
        print("[CoT] Response received.")

    parsed = parse_final_grid(resp.text)
    violations = []
    correct = False
    if parsed is None:
        violations.append("Could not parse a complete 4x4 grid from the model's output.")
    else:
        violations = full_grid_violations(parsed)
        correct = len(violations) == 0

    return CoTResult(
        prompt=prompt,
        transcript=resp.text,
        backend_name=resp.backend_name,
        parsed_grid=parsed,
        violations=violations,
        correct=correct,
    )


if __name__ == "__main__":
    from llm_backend import MockBackend

    result = run_cot(MockBackend())
    print("=== CoT transcript ===")
    print(result.transcript)
    print("\n=== Verdict ===")
    print("Correct:", result.correct)
    if result.violations:
        print("Violations found:")
        for v in result.violations:
            print(" -", v)
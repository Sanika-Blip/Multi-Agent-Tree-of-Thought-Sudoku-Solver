"""
baseline_ta1.py
---------------
Runs the FROZEN TA1 code (../ta1, imported unchanged) on one puzzle with
the real qwen2.5:3b, so evaluate.py can compare TA1 and TA2 on identical
inputs. Called as a subprocess so TA1's modules never mix with TA2's.

  python baseline_ta1.py '[[2,0,0,0],[0,0,2,3],[0,0,0,0],[1,0,0,4]]'

Last stdout line is 'RESULT_JSON {...}'.
"""

import json
import os
import sys
import time

TA1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ta1")
sys.path.insert(0, os.path.abspath(TA1_DIR))

from llm_backend import OllamaBackend          # noqa: E402  (TA1 modules)
from cot_solver import run_cot                 # noqa: E402
from tot_solver import run_tot                 # noqa: E402
from main import _OllamaCandidateAdapter       # noqa: E402
from sudoku import grid_to_str                 # noqa: E402


class CountingBackend(OllamaBackend):
    calls = 0

    def generate(self, prompt, temperature=0.2, timeout=300):
        CountingBackend.calls += 1
        return super().generate(prompt, temperature=temperature, timeout=timeout)


def main():
    puzzle = json.loads(sys.argv[1])
    backend = CountingBackend(model="qwen2.5:3b")

    # TA1 has no failure handling of its own, so a crash (e.g. the 300 s
    # Ollama timeout) is caught HERE and recorded as a failed case.
    t0 = time.time()
    cot, cot_error = None, None
    try:
        cot = run_cot(backend, puzzle)
        print("=== TA1 CoT ===\n" + cot.transcript + f"\ncorrect={cot.correct} violations={len(cot.violations)}")
    except Exception as e:
        cot_error = f"{type(e).__name__}: {e}"
        print(f"=== TA1 CoT ===\nCRASHED: {cot_error}")
    cot_s = time.time() - t0

    CountingBackend.calls = 0
    t0 = time.time()
    tot, tot_error = None, None
    try:
        tot = run_tot(_OllamaCandidateAdapter(backend), puzzle, max_candidates=3)
        print("\n=== TA1 ToT ===\n" + "\n".join(tot.log_lines))
        print(grid_to_str(tot.solved_grid) if tot.solved_grid else "NO SOLUTION")
    except Exception as e:
        tot_error = f"{type(e).__name__}: {e}"
        print(f"\n=== TA1 ToT ===\nCRASHED: {tot_error}")
    tot_s = time.time() - t0

    print("RESULT_JSON " + json.dumps({
        "cot_correct": bool(cot and cot.correct),
        "cot_violations": len(cot.violations) if cot else None,
        "cot_error": cot_error, "cot_seconds": round(cot_s, 2),
        "tot_solved": bool(tot and tot.solved_grid), "tot_error": tot_error,
        "tot_llm_calls": CountingBackend.calls,
        "tot_explored": tot.nodes_explored if tot else None,
        "tot_pruned": tot.nodes_pruned if tot else None,
        "tot_backtracks": tot.backtracks if tot else 0, "tot_seconds": round(tot_s, 2),
        "tot_grid": tot.solved_grid if tot else None,
    }))


if __name__ == "__main__":
    main()

"""
llm_backend.py
--------------
Wraps the actual model call so cot_solver.py / tot_solver.py don't care
whether they're talking to a real local Ollama model or a scripted mock.

Two backends:
  - OllamaBackend : calls http://localhost:11434/api/generate with
                    model="qwen2.5:3b" (or whatever you pass via --model).
  - MockBackend   : fully offline, deterministic. Used automatically if
                    Ollama isn't reachable, so the code (and the jury demo)
                    still runs end-to-end on a machine/network without
                    Ollama installed. Clearly labelled in all output.

get_backend() auto-detects which one to use.
"""

import json
import random
import urllib.request
import urllib.error
from dataclasses import dataclass


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class LLMResponse:
    text: str
    backend_name: str


class OllamaBackend:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            urllib.request.urlopen(req, timeout=1.5)
            return True
        except Exception:
            return False

    def generate(self, prompt: str, temperature: float = 0.2, timeout: int = 300) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return LLMResponse(text=body.get("response", ""), backend_name=f"ollama:{self.model}")


class MockBackend:
    """
    Deterministic stand-in for qwen2.5:3b, used only when Ollama is not
    reachable (e.g. no GPU/network on the demo machine). It reproduces the
    two behaviours this assignment asks you to document:

      1. A single greedy Chain-of-Thought pass that fills the grid
         left-to-right and, in doing so, paints itself into a corner on at
         least one cell (a realistic small-model failure mode -- it does
         not look ahead, so it happily repeats a value in a box or column
         it "hasn't checked yet").
      2. A Tree-of-Thought pass that proposes 2-3 candidates per cell, some
         of which are deliberately wrong first guesses, so the search log
         contains genuine pruned/dead-end branches for the viva.

    This backend NEVER consults the true solution to decide what's
    "correct" -- it only knows the rules, exactly like a real LLM would.
    Randomness is seeded for reproducible transcripts.
    """

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def is_available(self) -> bool:
        return True

    # -- Chain-of-Thought: one shot, fills the whole grid, greedy -------
    def generate_cot(self, grid) -> LLMResponse:
        """
        Simulates a single-shot Chain-of-Thought pass the way a small LLM
        actually does it: reason left-to-right, top-to-bottom, one cell at
        a time, checking the CURRENT board before each placement -- but
        with NO ability to look ahead and NO ability to undo an earlier
        cell once a later one turns out to be unsolvable.

        On this puzzle (only 5 givens, genuinely under-constrained early
        on) that greedy commitment is enough on its own to walk into a
        cell with zero legal values -- no scripted "bug" is injected here.
        When that happens the transcript documents it honestly and forces
        a value anyway, which is exactly the failure mode single-shot CoT
        cannot recover from.

        NOTE: this MockBackend path only runs when Ollama isn't reachable.
        Wired to the real qwen2.5:3b, run_cot() sends the prompt straight
        to the model and grades whatever it actually returns -- a real
        model might fail earlier, later, or not at all; that's for you to
        observe and report in the viva.
        """
        from sudoku import clone, is_valid_placement, N

        g = clone(grid)
        reasoning_lines = []

        for r in range(N):
            for c in range(N):
                if g[r][c] != 0:
                    continue
                placed = None
                for v in range(1, 5):
                    ok, _ = is_valid_placement(g, r, c, v)
                    if ok:
                        placed = v
                        break
                if placed is None:
                    placed = 1
                    reasoning_lines.append(
                        f"Cell ({r},{c}): every value 1-4 conflicts with what's "
                        f"already on the board, but a single pass has no way to "
                        f"undo an earlier step, so forcing {placed} and "
                        f"continuing."
                    )
                else:
                    reasoning_lines.append(
                        f"Cell ({r},{c}): {placed} doesn't conflict with the "
                        f"cells filled so far, placing it."
                    )
                g[r][c] = placed
        reasoning_lines.append("Final grid:")
        for row in g:
            reasoning_lines.append(" ".join(str(x) for x in row))
        return LLMResponse(text="\n".join(reasoning_lines), backend_name="mock:qwen2.5:3b(offline-sim)")

    # -- Tree-of-Thought: candidates for ONE cell, with a genuine wrong
    #    guess mixed in some of the time, so pruning is real. -----------
    def propose_candidates(self, grid, r, c, k=3):
        from sudoku import is_valid_placement

        valid = [v for v in range(1, 5) if is_valid_placement(grid, r, c, v)[0]]
        # Build a candidate list of size <=k: all truly valid values, plus
        # (sometimes) one plausible-looking but invalid value first, to
        # simulate the model "guessing" before checking.
        candidates = []
        if self.rng.random() < 0.5 and len(valid) < 4:
            invalid_pool = [v for v in range(1, 5) if v not in valid]
            if invalid_pool:
                candidates.append(self.rng.choice(invalid_pool))
        candidates.extend(valid)
        # de-dup, keep order, cap at k
        seen = set()
        ordered = []
        for v in candidates:
            if v not in seen:
                seen.add(v)
                ordered.append(v)
        ordered = ordered[:k] if len(ordered) > k else ordered
        if not ordered:
            ordered = [1, 2, 3][:k]
        return ordered
"""
agents.py  (U4-1 role design, U3-4 failure handling, U4-3 guard)
---------
The three worker agents of the TA2 solver. The Supervisor that routes
between them lives in graph.py; the Voter lives in vote.py.

  Proposer     (LLM, qwen2.5:3b)  suggests candidate digits for one cell.
  RuleChecker  (deterministic)    guard agent: screens untrusted notes for
                                  prompt injection and rejects every
                                  candidate that breaks a rule or is a
                                  known dead end. It is the ONLY agent
                                  that can prune a branch.
  Backtracker  (deterministic)    undoes the last placement at a dead end
                                  and writes the dead end to long-term
                                  memory.
"""

import re

from langchain_ollama import ChatOllama

from sudoku import N, is_valid_placement, grid_to_str, board_key
from memory import is_known_dead_end, remember_dead_end

ROLE_SPECS = {
    "Supervisor": {
        "goal": "Route each hop to the right agent until the board is solved, proven unsolvable, or MAX_HOPS is reached.",
        "tools": "none (reads state only)",
        "inputs/outputs": "in: board, pending candidates, stack, hop count -> out: next agent name",
        "limits": "Hard stop at MAX_HOPS; never edits the board itself.",
    },
    "Proposer": {
        "goal": "Suggest up to 3 candidate digits for the next empty cell, most likely first.",
        "tools": "qwen2.5:3b via ChatOllama; RAG hint from Chroma; dead-end list from the Store",
        "inputs/outputs": "in: board, cell, hint, notes, known dead ends -> out: list of digits",
        "limits": "Cannot place digits or prune; its output is untrusted. Retries once on failure, then hands over to the fallback.",
    },
    "RuleChecker": {
        "goal": "Guard: accept only legal, not-known-dead candidates; strip prompt injection from notes.",
        "tools": "is_valid_placement(); long-term dead-end Store; injection regex",
        "inputs/outputs": "in: board, cell, proposed digits -> out: accepted digits, pruned digits with reasons",
        "limits": "Never invents strategy; only fills in legal digits the model missed (flagged as fallback).",
    },
    "Backtracker": {
        "goal": "At a dead end, undo the most recent placement and resume with its next sibling.",
        "tools": "DFS stack in graph state; long-term dead-end Store (write)",
        "inputs/outputs": "in: stack -> out: restored board, remaining siblings; dead end saved",
        "limits": "Only records a dead end after its whole subtree is exhausted, never on a hop-limit stop.",
    },
    "Voter": {
        "goal": "Combine 3 independent solver runs into one answer by majority vote.",
        "tools": "full_grid_violations() for a final rule check",
        "inputs/outputs": "in: 3 final boards + status -> out: winning board, agreement count",
        "limits": "Needs at least 2 agreeing valid boards; otherwise reports no majority.",
    },
}

PROPOSER_PROMPT = """You are the Proposer in a 4x4 mini-Sudoku solver. Digits 1-4, 2x2 boxes, '.' = empty.
Guidance: {hint}

Board:
{board}

User notes: {notes}
Known dead ends for this cell (do not propose): {avoid}

Propose up to {k} digits for the empty cell at row {r}, column {c} (0-indexed), most likely first.
Reply with ONLY the digits, comma-separated."""


class InjectedFailure(RuntimeError):
    pass


class Proposer:
    def __init__(self, model="qwen2.5:3b", temperature=0.4, hint="", fail_on_calls=()):
        self.llm = ChatOllama(model=model, temperature=temperature, num_predict=24)
        self.hint = hint
        self.fail_on_calls = set(fail_on_calls)   # for the U3-4 failure demo
        self.calls = 0

    def _call(self, prompt: str) -> str:
        self.calls += 1
        if self.calls in self.fail_on_calls:
            raise InjectedFailure("injected fault: Ollama did not answer (simulated timeout)")
        return self.llm.invoke(prompt).content

    def propose(self, grid, cell, notes="", avoid=(), k=3):
        """Returns (digits, events). digits == [] means the model gave us
        nothing usable; the RuleChecker's fallback then takes over."""
        r, c = cell
        prompt = PROPOSER_PROMPT.format(hint=self.hint, board=grid_to_str(grid),
                                        notes=notes or "(none)",
                                        avoid=", ".join(map(str, avoid)) or "(none)",
                                        k=k, r=r, c=c)
        events = []
        for attempt in (1, 2):
            try:
                text = self._call(prompt)
            except Exception as e:  # timeout, connection refused, injected fault...
                events.append(f"[failure] Proposer call #{self.calls} (attempt {attempt}): "
                              f"{type(e).__name__}: {e}")
                continue
            digits = []
            for d in re.findall(r"\d", text):   # keep ANY digit so the guard sees bad ones
                if int(d) not in digits:
                    digits.append(int(d))
            if digits:
                return digits[:k], events
            events.append(f"[failure] Proposer reply had no digits: {text!r}")
        events.append("[recovery] Proposer gave nothing usable after 1 retry -> "
                      "RuleChecker fallback will enumerate legal digits.")
        return [], events


class RuleChecker:
    INJECTION = re.compile(r"ignore|override|disregard|system prompt|new rules|"
                           r"rules (have )?changed|always (answer|place|use|put)", re.I)

    def __init__(self, enabled=True, store=None, user_id="demo_user", use_memory=True):
        self.enabled = enabled
        self.store = store
        self.user_id = user_id
        self.use_memory = use_memory and store is not None

    def screen_notes(self, notes: str):
        if self.enabled and notes and self.INJECTION.search(notes):
            return "(removed by guard: possible prompt injection)", \
                f"[guard] notes blocked as prompt injection: {notes!r}"
        return notes, None

    def known_dead(self, grid, cell, value) -> bool:
        return self.use_memory and is_known_dead_end(self.store, board_key(grid), cell, value,
                                                     self.user_id)

    def check(self, grid, cell, proposed):
        """Returns (accepted, pruned, fallback) where pruned is [(digit, reason)]."""
        r, c = cell
        if not self.enabled:   # unguarded baseline for the injection demo
            return [v for v in proposed if 1 <= v <= N], [], []
        accepted, pruned = [], []
        for v in proposed:
            if not 1 <= v <= N:
                pruned.append((v, f"out of range (must be 1-{N})"))
                continue
            ok, reason = is_valid_placement(grid, r, c, v)
            if not ok:
                pruned.append((v, f"invalid: {reason}"))
            elif self.known_dead(grid, cell, v):
                pruned.append((v, "known dead end (long-term memory)"))
            else:
                accepted.append(v)
        # Same safety net as TA1: legal digits the model missed are added
        # last and flagged, so a model omission cannot sink the search.
        fallback = [v for v in range(1, N + 1)
                    if v not in proposed and is_valid_placement(grid, r, c, v)[0]
                    and not self.known_dead(grid, cell, v)]
        return accepted + fallback, pruned, fallback


class Backtracker:
    def __init__(self, store=None, user_id="demo_user", use_memory=True):
        self.store = store
        self.user_id = user_id
        self.use_memory = use_memory and store is not None

    def record(self, frame) -> bool:
        if self.use_memory:
            remember_dead_end(self.store, board_key(frame["grid"]), frame["cell"],
                              frame["value"], self.user_id)
        return self.use_memory

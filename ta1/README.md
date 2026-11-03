# Sudoku Mini-Puzzle Solver: Chain-of-Thought vs Tree-of-Thought

Reasoning-framework deep-dive for the Agentic AI jury. Same 4x4 Sudoku
puzzle, solved two ways: a single Chain-of-Thought (CoT) pass, and a
Tree-of-Thought (ToT) search with explicit candidate generation,
validity-checking, and backtracking.

## The puzzle

```
2 . | . .
. . | 2 3
-----------
. . | . .
1 . | . 4
```

5 givens, verified (by brute force, `sudoku.py`) to have exactly **one**
solution. It is deliberately under-constrained early on so that a greedy,
no-lookahead pass can commit to a locally-valid cell that turns out to be
globally wrong several cells later -- which is the whole point of the demo.

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # just matplotlib

# Optional, for the REAL LLM instead of the offline mock:
ollama pull qwen2.5:3b
ollama serve   # if not already running as a service
```

## Running it

```bash
python3 main.py                      # auto-detects Ollama; falls back to mock
python3 main.py --model qwen2.5:3b   # be explicit about the model tag
python3 main.py --force-mock         # always use the offline mock
```

Each run prints Steps 2-5 to the console and writes every deliverable into
`outputs/`:

| File | What it is |
|---|---|
| `cot_transcript.txt` | Full prompt + single-shot CoT transcript + verdict |
| `tot_search_log.txt` | Step-by-step ToT search log (every candidate tried, pruned, or backtracked) |
| `tot_tree.json` | The complete search tree, machine-readable (every node: cell, value, valid/pruned/backtracked) |
| `search_tree.png` | The search tree diagram -- **this is the "documented search tree diagram" deliverable** |
| `comparison.md` | Step 5's written comparison and conclusion |

## How it's built (Methodology)

1. **`sudoku.py`** -- board representation, row/column/box validity checker,
   and a plain backtracking ground-truth solver used only to confirm the
   puzzle is uniquely solvable before handing it to any model.
2. **`llm_backend.py`** -- abstracts the model call. `OllamaBackend` talks
   to a local Ollama server running `qwen2.5:3b`. `MockBackend` is a fully
   offline, deterministic stand-in used automatically if Ollama isn't
   reachable, so the demo always runs (e.g. on a jury-room laptop with no
   GPU/network) -- every output file clearly labels which backend produced
   it.
3. **`cot_solver.py`** -- sends **one** prompt asking the model to reason
   cell-by-cell and output a finished grid. No tool calls, no
   verification loop, no way to undo a step. Whatever grid comes back is
   graded against the actual Sudoku rules.
4. **`tot_solver.py`** -- at each empty cell (impasse), asks the model for
   2-3 candidate digits, checks each candidate against the rules
   immediately, descends into valid ones, and backtracks when a branch
   dead-ends further down. Every node -- kept, pruned, or backtracked -- is
   logged into a tree structure.
5. **`tree_viz.py`** -- renders that tree to `search_tree.png`: green nodes
   are on the winning path, grey nodes were valid but backtracked out of
   (a dead end further down), red nodes were pruned immediately for a rule
   violation.
6. **`main.py`** -- runs both solvers back-to-back and writes the
   comparison.

## What actually happened (offline mock run, for reference)

- **CoT**: fails at the very first impasse (cell (0,3)) -- every digit
  1-4 already conflicts with earlier greedy placements, but a single pass
  can't undo them, so it forces a value and the error cascades: **10 rule
  violations** in the final grid.
- **ToT**: 35 candidate placements explored, **19 pruned** for rule
  violations, **5 genuine backtracks** out of multi-level dead ends,
  solves the puzzle correctly.

Re-running against the real `qwen2.5:3b` via Ollama may fail earlier,
later, or occasionally not at all -- report whatever you actually observe;
`main.py` grades the model's real output either way, it does not force a
particular result.

## Likely jury questions -- where to point

**"Show me one specific dead-end branch you pruned and explain why it was invalid."**
Open `search_tree.png` or `tot_search_log.txt`. Example: at depth 2, cell
(0,3) with the model having placed 1 at (0,1) and 4 at (0,2), *every*
remaining candidate for (0,3) conflicts with row 0 -- the branch dead-ends
and the tree backtracks all the way up to try `(0,1)=3` instead. `tot_tree.json`
has the same node with `"status": "backtracked"` plus its pruned children.

**"Why does Sudoku benefit from search/backtracking more than simple arithmetic does?"**
Arithmetic is *compositional and local* -- each step's correctness doesn't
depend on choices made many steps earlier or later, so a single forward
pass is sound. Sudoku is a *constraint-satisfaction problem*: a cell's
correct value depends on the entire row, column, and box, including cells
filled both before and after it in solving order. A locally-valid choice
can silently make a *different, later* cell unsolvable, and there is no
way to detect that without either looking ahead or being willing to undo
a past step -- which is exactly what backtracking search provides and a
single CoT pass does not.

## Implementation plan (Step 7, for the deck)

- **Now (prototype):** local Python + Ollama, single puzzle instance, tree
  rendered to PNG for the demo.
- **Next:** swap the raw HTTP calls in `llm_backend.py` for a proper
  agent framework (LangGraph is a natural fit -- each ToT node maps to a
  graph node with a conditional edge for "valid -> descend" vs
  "invalid -> prune/backtrack") so the search itself is orchestrated by
  the framework rather than hand-rolled recursion.
- **Scaling up:** generalize `sudoku.py`'s box size to run true 9x9
  boards, and swap the brute-force ground-truth checker for a constraint
  library (e.g. `python-constraint`) once tree sizes get too large to
  document by hand.
- **Deployment:** wrap `main.py` behind a small FastAPI endpoint that
  accepts a puzzle and returns the transcript + tree JSON, so a front end
  can visualize the search live instead of reading static files.

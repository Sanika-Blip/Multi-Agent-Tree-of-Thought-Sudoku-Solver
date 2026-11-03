# CoT vs ToT -- Written Comparison

**Chain-of-Thought (single shot):** FAILED.
- 7 rule violation(s) in the final grid (see cot_transcript.txt).
- Root cause: no mechanism to look ahead or undo an earlier cell once a later cell turns out to be unsolvable.

**Tree-of-Thought (search + backtracking):** SOLVED correctly.
- Explored 36 candidate placements, pruned 23 immediately for rule violations, and backtracked 2 time(s) out of dead ends several moves deep.
- Every candidate is validated against row/column/box rules before being committed, and any branch that later dead-ends is undone -- which is exactly the mechanism single-shot CoT lacks.

**Conclusion:** Tree-of-Thought solved the puzzle correctly; single-shot Chain-of-Thought did not, because Sudoku is a constraint-satisfaction problem where an early, locally-reasonable choice can make a *later* cell unsolvable. Only a method that can check constraints at each step and backtrack when a branch dies can guarantee a correct fill; a single forward pass has no way to detect or repair that until it's already too late.
## Rules of 4x4 mini-Sudoku
A 4x4 mini-Sudoku has 4 rows, 4 columns and four 2x2 boxes. Every row, every column and every 2x2 box must contain the digits 1, 2, 3 and 4 exactly once. Box 0 is rows 0-1 and columns 0-1, box 1 is rows 0-1 and columns 2-3, box 2 is rows 2-3 and columns 0-1, box 3 is rows 2-3 and columns 2-3.

## Naked single
A naked single is an empty cell where only one digit is still legal after removing every digit already present in its row, its column and its box. A naked single can be placed immediately with no guessing and no risk of a later dead end.

## Hidden single
A hidden single is a digit that has only one legal position left inside a row, a column or a box, even if that cell still has other legal digits. Placing hidden singles first shrinks the search tree because it removes branching.

## Candidate generation for one cell
To propose candidates for an empty cell, list the digits 1-4, cross out digits that appear in the same row, the same column and the same 2x2 box, and propose the digits that remain. Proposing a digit that is already in the row, column or box is always wrong and will be pruned by the rule check.

## Minimum remaining values (MRV)
The minimum remaining values heuristic picks the empty cell with the fewest legal candidates next. On 4x4 puzzles MRV usually removes most backtracking because forced cells are filled before guesses are made. Plain left-to-right order is simpler to explain but explores more dead ends.

## Backtracking and dead ends
A dead end is a partial board where some empty cell has zero legal digits. When a dead end is reached, the solver must undo the most recent guess and try the next candidate for that cell. If every candidate for that cell also fails, it undoes one level further up. This undo step is called backtracking.

## Why single-pass Chain-of-Thought fails on Sudoku
Sudoku is a constraint-satisfaction problem. A digit that is legal now can make a different cell impossible several moves later. A single forward pass, such as one Chain-of-Thought prompt, cannot undo an earlier placement, so it either forces an illegal digit or leaves the board inconsistent.

## Worked example: TA1 jury puzzle
The TA1 puzzle has givens 2 at (0,0), 2 at (1,2), 3 at (1,3), 1 at (3,0) and 4 at (3,3). Its unique solution is row 0: 2 3 4 1, row 1: 4 1 2 3, row 2: 3 4 1 2, row 3: 1 2 3 4. A known dead end: placing 1 at (0,1) and then 4 at (0,2) leaves cell (0,3) with no legal digit, so the solver backtracks to try 3 at (0,1).

## Why the rule checker, not the model, decides validity
A 3B language model often proposes digits that break a rule because it does not reliably track all row, column and box constraints at once. The rule checker is deterministic code: it checks the row, column and box of the cell and rejects any conflict. The model only suggests the order in which to try digits.

## Voting across independent solvers
Running several independent solvers and taking the majority answer reduces the chance that a single unlucky run, such as one that hits its hop limit or receives a broken model reply, decides the final result. Voting costs roughly one extra full solve per extra solver.

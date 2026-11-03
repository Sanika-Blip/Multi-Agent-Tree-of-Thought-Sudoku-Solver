"""
vote.py  (topic row: 3 independent solvers in parallel + vote)
-------
Runs 3 solvers with different sampling temperatures on separate threads
(and separate thread_ids). They do NOT share the dead-end memory, so the
runs are independent. The Voter takes the majority board among the runs
that finished with a valid board.
"""

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from rag import get_vectorstore, hint_for_cell
from solver import solve, new_thread_id
from sudoku import board_key, full_grid_violations

TEMPERATURES = (0.2, 0.7, 1.0)


def run_vote(puzzle_id, max_hops=120, temperatures=TEMPERATURES, verbose=False):
    hint = hint_for_cell(get_vectorstore(), k=1)
    run_id = new_thread_id(f"vote-{puzzle_id}")

    def one(i_t):
        i, t = i_t
        return solve(puzzle_id, f"{run_id}-s{i}", max_hops=max_hops, temperature=t,
                     use_memory=False, hint=hint, verbose=verbose, tag=f"[solver {i}] ")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(temperatures)) as pool:
        results = list(pool.map(one, enumerate(temperatures, 1)))
    wall = round(time.time() - t0, 2)

    ballots = [board_key(r["grid"]) for r in results
               if r["status"] == "solved" and not full_grid_violations(r["grid"])]
    tally = Counter(ballots)
    winner, votes = tally.most_common(1)[0] if tally else (None, 0)
    majority = votes >= len(temperatures) // 2 + 1
    rows = [{"solver": i, "temperature": t, "status": r["status"], "hops": r["hops"],
             "llm_calls": r["stats"]["llm_calls"], "pruned": r["stats"]["pruned_rule"],
             "backtracks": r["stats"]["backtracks"], "seconds": r["wall_seconds"],
             "board": board_key(r["grid"])}
            for i, (t, r) in enumerate(zip(temperatures, results), 1)]
    return {"rows": rows, "winner": winner if majority else None, "votes": votes,
            "majority": majority, "wall_seconds": wall,
            "llm_calls": sum(r["llm_calls"] for r in rows),
            "logs": [r["log"] for r in results]}


def print_vote(v):
    print(f"{'solver':>6} {'temp':>5} {'status':>10} {'hops':>5} {'calls':>6} "
          f"{'pruned':>7} {'backtr':>7} {'sec':>6}  board")
    for r in v["rows"]:
        print(f"{r['solver']:>6} {r['temperature']:>5} {r['status']:>10} {r['hops']:>5} "
              f"{r['llm_calls']:>6} {r['pruned']:>7} {r['backtracks']:>7} {r['seconds']:>6}  {r['board']}")
    print(f"VOTE: {'majority ' + str(v['votes']) + '/' + str(len(v['rows'])) if v['majority'] else 'NO MAJORITY'}"
          f" -> {v['winner']}  | total LLM calls {v['llm_calls']}, wall {v['wall_seconds']}s")

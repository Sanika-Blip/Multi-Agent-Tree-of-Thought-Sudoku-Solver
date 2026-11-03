"""
evaluate.py  (EV + U4-4 cost measurement)
-----------
Runs the same 5 puzzles through:
  TA1 : frozen CoT + frozen ToT (baseline_ta1.py, subprocess)
  TA2 : single multi-agent solver, cold dead-end memory
  TA2 : 3-solver vote
  TA2 : single solver again, warm dead-end memory (memory written by the
        cold pass is read back from SQLite)
Writes traces/ta1_Pn.txt, traces/ta2_Pn.txt, traces/ta2_vote_Pn.txt,
traces/eval_results.json and traces/eval_table.md.
"""

import contextlib
import io
import json
import os
import subprocess
import sys

from memory import get_store, clear_dead_ends
from rag import get_vectorstore, hint_for_cell
from solver import solve, new_thread_id
from sudoku import PUZZLES, grid_to_str, brute_force_solutions
from vote import run_vote, print_vote

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "..", "traces")


def run_ta1(pid):
    out = subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "baseline_ta1.py"),
                          json.dumps(PUZZLES[pid])], capture_output=True, text=True,
                         encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    with open(os.path.join(TRACES, f"ta1_{pid}.txt"), "w", encoding="utf-8") as f:
        f.write(out.stdout + out.stderr)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT_JSON ")][-1]
    return json.loads(line[len("RESULT_JSON "):])


def run_ta2(pid, hint, label):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Puzzle {pid} ({label}):\n{grid_to_str(PUZZLES[pid])}\n")
        r = solve(pid, new_thread_id(f"eval-{label}-{pid}"), hint=hint)
        print(f"\nstatus={r['status']} hops={r['hops']} wall={r['wall_seconds']}s stats={r['stats']}")
    name = f"ta2_{pid}.txt" if label == "cold" else f"ta2_warm_{pid}.txt"
    with open(os.path.join(TRACES, name), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    truth = brute_force_solutions(PUZZLES[pid], 1)[0]
    return {"solved": r["status"] == "solved" and r["grid"] == truth, "hops": r["hops"],
            "seconds": r["wall_seconds"], **r["stats"]}


def run_ta2_vote(pid):
    v = run_vote(pid)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_vote(v)
        for i, log in enumerate(v["logs"], 1):
            print(f"\n--- solver {i} log ---\n" + "\n".join(log))
    with open(os.path.join(TRACES, f"ta2_vote_{pid}.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    truth = "".join(str(x) for row in brute_force_solutions(PUZZLES[pid], 1)[0] for x in row)
    single_ok = sum(r["status"] == "solved" for r in v["rows"])
    return {"majority": v["majority"], "correct": v["winner"] == truth, "votes": v["votes"],
            "solvers_solved": single_ok, "llm_calls": v["llm_calls"], "seconds": v["wall_seconds"]}


def table(res):
    yn = lambda b: "yes" if b else "NO"  # noqa: E731
    lines = ["| Case | TA1 CoT correct | TA1 ToT solved | TA1 ToT calls | TA1 ToT backtracks | TA1 ToT s "
             "| TA2 solved | TA2 calls | TA2 hops | TA2 backtracks | TA2 s | Vote correct (votes) | Vote calls | Vote s "
             "| TA2 warm calls | TA2 warm backtracks |",
             "|" + "---|" * 16]
    for pid, r in res.items():
        a, b, v, w = r["ta1"], r["ta2"], r["vote"], r["warm"]
        lines.append(f"| {pid} | {yn(a['cot_correct'])} | {yn(a['tot_solved'])} | {a['tot_llm_calls']} | "
                     f"{a['tot_backtracks']} | {a['tot_seconds']} | {yn(b['solved'])} | {b['llm_calls']} | "
                     f"{b['hops']} | {b['backtracks']} | {b['seconds']} | {yn(v['correct'])} ({v['votes']}/3) | "
                     f"{v['llm_calls']} | {v['seconds']} | {w['llm_calls']} | {w['backtracks']} |")
    tot = lambda f: sum(f(r) for r in res.values())  # noqa: E731
    lines.append(f"| **Total** | {tot(lambda r: r['ta1']['cot_correct'])}/5 | {tot(lambda r: r['ta1']['tot_solved'])}/5 | "
                 f"{tot(lambda r: r['ta1']['tot_llm_calls'])} | {tot(lambda r: r['ta1']['tot_backtracks'])} | "
                 f"{round(tot(lambda r: r['ta1']['tot_seconds']), 1)} | {tot(lambda r: r['ta2']['solved'])}/5 | "
                 f"{tot(lambda r: r['ta2']['llm_calls'])} | {tot(lambda r: r['ta2']['hops'])} | "
                 f"{tot(lambda r: r['ta2']['backtracks'])} | {round(tot(lambda r: r['ta2']['seconds']), 1)} | "
                 f"{tot(lambda r: r['vote']['correct'])}/5 | {tot(lambda r: r['vote']['llm_calls'])} | "
                 f"{round(tot(lambda r: r['vote']['seconds']), 1)} | {tot(lambda r: r['warm']['llm_calls'])} | "
                 f"{tot(lambda r: r['warm']['backtracks'])} |")
    return "\n".join(lines)


def main():
    os.makedirs(TRACES, exist_ok=True)
    print(f"cleared {clear_dead_ends(get_store())} dead ends -> cold start", flush=True)
    hint = hint_for_cell(get_vectorstore(), k=1)
    res = {}
    for pid in PUZZLES:
        print(f"[{pid}] TA1 ...", flush=True)
        res[pid] = {"ta1": run_ta1(pid)}
        print(f"[{pid}] TA2 single ...", flush=True)
        res[pid]["ta2"] = run_ta2(pid, hint, "cold")
        print(f"[{pid}] TA2 vote ...", flush=True)
        res[pid]["vote"] = run_ta2_vote(pid)
    for pid in PUZZLES:   # second pass: dead ends from the cold pass are now in SQLite
        print(f"[{pid}] TA2 warm ...", flush=True)
        res[pid]["warm"] = run_ta2(pid, hint, "warm")
    with open(os.path.join(TRACES, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    md = table(res)
    with open(os.path.join(TRACES, "eval_table.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()

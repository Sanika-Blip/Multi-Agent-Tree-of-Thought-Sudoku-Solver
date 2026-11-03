"""
main.py -- TA2 command line

  python main.py solve P1 [--thread T] [--max-hops N] [--resume]
                          [--fail-on 3,4] [--notes "..."] [--no-guard] [--no-memory]
  python main.py ask T "question about the solve on thread T"
  python main.py vote P1
  python main.py rag "question" [--k 1]
  python main.py profile [--set key=value]
  python main.py memory            # how many dead ends are stored
"""

import argparse
import os

from sudoku import PUZZLES, grid_to_str

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def cmd_solve(a):
    from solver import solve
    from search_tree import save_tree
    fail = tuple(int(x) for x in a.fail_on.split(",")) if a.fail_on else ()
    print(f"Puzzle {a.puzzle}:\n{grid_to_str(PUZZLES[a.puzzle])}\n")
    final = solve(a.puzzle, a.thread, max_hops=a.max_hops, resume=a.resume,
                  fail_on_calls=fail, notes=a.notes, guard=not a.no_guard,
                  use_memory=not a.no_memory, temperature=a.temperature)
    print(f"\nthread_id={final['thread_id']}  status={final['status']}  hops={final['hops']}  "
          f"wall={final['wall_seconds']}s")
    print(f"stats={final['stats']}")
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"tree_{final['thread_id']}")
    save_tree(final["nodes"], stem + ".png", stem + ".json",
              title=f"TA2 search tree - {a.puzzle} ({final['status']})")
    print(f"tree saved: {stem}.png")


def cmd_ask(a):
    from solver import ask
    reply, record = ask(a.thread, a.question)
    print(f"[thread {a.thread}] record given to the model:\n{record}\n")
    print(f"Q: {a.question}\nA: {reply}")


def cmd_vote(a):
    from vote import run_vote, print_vote
    print_vote(run_vote(a.puzzle, max_hops=a.max_hops))


def cmd_rag(a):
    from rag import get_vectorstore, answer
    reply, titles = answer(get_vectorstore(), a.question, k=a.k)
    print(f"k={a.k} retrieved: {titles}\nQ: {a.question}\nA: {reply}")


def cmd_profile(a):
    from memory import get_store, get_profile, set_profile, DEFAULT_USER
    store = get_store()
    if a.set:
        key, _, val = a.set.partition("=")
        set_profile(store, **{key: val})
    print(f"profile ({DEFAULT_USER}): {get_profile(store)}")


def cmd_memory(a):
    from memory import get_store, count_dead_ends, DEFAULT_USER
    print(f"dead ends stored for {DEFAULT_USER}: {count_dead_ends(get_store())}")


def main():
    p = argparse.ArgumentParser(description="TA2 multi-agent ToT Sudoku solver")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("puzzle", choices=sorted(PUZZLES))
    s.add_argument("--thread")
    s.add_argument("--max-hops", type=int, default=120)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--fail-on", default="", help="Proposer call numbers that fail, e.g. 3,4")
    s.add_argument("--notes", default="")
    s.add_argument("--no-guard", action="store_true")
    s.add_argument("--no-memory", action="store_true")
    s.add_argument("--temperature", type=float, default=0.4)
    s.set_defaults(fn=cmd_solve)
    s = sub.add_parser("ask")
    s.add_argument("thread")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("vote")
    s.add_argument("puzzle", choices=sorted(PUZZLES))
    s.add_argument("--max-hops", type=int, default=120)
    s.set_defaults(fn=cmd_vote)
    s = sub.add_parser("rag")
    s.add_argument("question")
    s.add_argument("--k", type=int, default=3)
    s.set_defaults(fn=cmd_rag)
    s = sub.add_parser("profile")
    s.add_argument("--set")
    s.set_defaults(fn=cmd_profile)
    s = sub.add_parser("memory")
    s.set_defaults(fn=cmd_memory)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

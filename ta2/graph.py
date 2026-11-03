"""
graph.py  (U4-2 coordination: supervisor with MAX_HOPS)
--------
LangGraph StateGraph. Every hop goes through the Supervisor, which routes
to exactly one worker agent:

    START -> supervisor -+-> proposer    -> supervisor
                         +-> checker     -> supervisor
                         +-> backtracker -> supervisor
                         +-> finish      -> END

The search is an explicit DFS whose stack lives in the graph state, so the
SqliteSaver checkpointer captures the whole search after every hop. That is
what lets a run stopped by MAX_HOPS be resumed on the same thread_id.
"""

import operator
import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from sudoku import N, clone, find_empty, full_grid_violations, grid_to_str, is_valid_placement

MAX_HOPS = 120


class SolveState(TypedDict, total=False):
    puzzle_id: str
    grid: list
    stack: list              # DFS frames: {grid, cell, value, rest, node, parent}
    pending: Optional[dict]  # {cell, candidates, checked, parent}
    nodes: list              # flat search tree, see search_tree.py
    current_node: int
    hops: int
    max_hops: int
    status: str              # running | solved | hop_limit | unsolvable | invalid
    next: str
    notes: str
    cell_order: str          # "first" or "mrv" (from the long-term profile)
    stats: dict
    log: Annotated[list, operator.add]
    route: Annotated[list, operator.add]


def initial_state(puzzle_id, puzzle, notes="", cell_order="first", max_hops=MAX_HOPS):
    return {
        "puzzle_id": puzzle_id, "grid": clone(puzzle), "stack": [], "pending": None,
        "nodes": [{"id": 0, "parent": None, "cell": None, "value": None,
                   "status": "root", "reason": "initial board"}],
        "current_node": 0, "hops": 0, "max_hops": max_hops, "status": "running",
        "notes": notes, "cell_order": cell_order,
        "stats": {"llm_calls": 0, "pruned_rule": 0, "pruned_memory": 0, "fallback_digits": 0,
                  "backtracks": 0, "failures": 0, "dead_ends_saved": 0, "illegal_accepted": 0, "llm_seconds": 0.0},
        "log": [], "route": [],
    }


def build_graph(proposer, checker, backtracker, checkpointer=None, verbose=True, tag=""):
    def say(lines):
        if verbose:
            for ln in lines:
                print(f"{tag}{ln}", flush=True)
        return lines

    def bump(state, **inc):
        stats = dict(state["stats"])
        for k, v in inc.items():
            stats[k] = stats.get(k, 0) + v
        return stats

    # ---------------- Supervisor ----------------
    def supervisor(state):
        hops = state["hops"] + 1
        pending = state.get("pending")
        if state["status"] != "running":
            nxt, why = "finish", f"status is {state['status']}"
        elif find_empty(state["grid"]) is None:
            nxt, why = "finish", "no empty cell left"
        elif hops > state["max_hops"]:
            nxt, why = "finish", f"MAX_HOPS={state['max_hops']} reached"
        elif pending is None:
            nxt, why = "proposer", "need candidates for the next empty cell"
        elif pending["candidates"] or not pending["checked"]:
            # Fresh proposals always go to the checker, even an empty list:
            # its fallback must get the chance to add legal digits before
            # the branch can be called a dead end.
            nxt, why = "checker", f"{len(pending['candidates'])} candidate(s) to check for {tuple(pending['cell'])}"
        else:
            nxt, why = "backtracker", f"no candidates left for {tuple(pending['cell'])}"
        return {"hops": hops, "next": nxt,
                "route": say([f"[hop {hops:3d}] supervisor -> {nxt:11s} ({why})"])}

    # ---------------- Proposer ----------------
    def proposer_node(state):
        t0 = time.time()
        grid = state["grid"]
        cell = find_empty(grid, strategy=state.get("cell_order", "first"))
        notes, guard_event = checker.screen_notes(state.get("notes", ""))
        avoid = [v for v in range(1, N + 1) if checker.known_dead(grid, cell, v)]
        calls_before = proposer.calls
        digits, events = proposer.propose(grid, cell, notes=notes, avoid=avoid)
        lines = ([guard_event] if guard_event else []) + events + \
            [f"    Proposer: cell {cell} -> {digits}" + (f" (memory says avoid {avoid})" if avoid else "")]
        return {"pending": {"cell": list(cell), "candidates": digits, "checked": False,
                            "parent": state["current_node"]},
                "stats": bump(state, llm_calls=proposer.calls - calls_before,
                              failures=sum(e.startswith("[failure]") for e in events),
                              llm_seconds=time.time() - t0),
                "log": say(lines)}

    # ---------------- RuleChecker (guard) ----------------
    def checker_node(state):
        pending, grid = state["pending"], state["grid"]
        cell = tuple(pending["cell"])
        nodes = list(state["nodes"])
        lines, stats = [], dict(state["stats"])
        if pending["checked"]:          # siblings restored by the Backtracker
            accepted = list(pending["candidates"])
        else:
            accepted, pruned, fallback = checker.check(grid, cell, pending["candidates"])
            for v, reason in pruned:
                nodes.append({"id": len(nodes), "parent": pending["parent"], "cell": list(cell),
                              "value": v, "status": "pruned", "reason": reason})
                key = "pruned_memory" if "memory" in reason else "pruned_rule"
                stats[key] += 1
                lines.append(f"    RuleChecker: {v}@{cell} PRUNED ({reason})")
            if fallback:
                stats["fallback_digits"] += len(fallback)
                lines.append(f"    RuleChecker: model missed legal digit(s) {fallback} -> added as fallback")
        if not accepted:
            lines.append(f"    RuleChecker: nothing legal left for {cell}")
            return {"pending": {**pending, "candidates": [], "checked": True},
                    "nodes": nodes, "stats": stats, "log": say(lines)}
        value, rest = accepted[0], accepted[1:]
        node_id = len(nodes)
        nodes.append({"id": node_id, "parent": pending["parent"], "cell": list(cell),
                      "value": value, "status": "valid", "reason": "passes row/col/box check"})
        legal, why = is_valid_placement(grid, cell[0], cell[1], value)
        if not legal:   # only possible with the guard switched off
            stats["illegal_accepted"] += 1
            nodes[node_id]["reason"] = f"ACCEPTED WITHOUT GUARD although {why}"
        new_grid = clone(grid)
        new_grid[cell[0]][cell[1]] = value
        frame = {"grid": grid, "cell": list(cell), "value": value, "rest": rest,
                 "node": node_id, "parent": pending["parent"]}
        lines.append(f"    RuleChecker: {value}@{cell} accepted, descending"
                     + (f" (siblings kept for later: {rest})" if rest else "")
                     + ("" if legal else f"  <-- ILLEGAL, {why} (guard off)"))
        return {"grid": new_grid, "stack": state["stack"] + [frame], "pending": None,
                "nodes": nodes, "current_node": node_id, "stats": stats, "log": say(lines)}

    # ---------------- Backtracker ----------------
    def backtracker_node(state):
        stack = list(state["stack"])
        if not stack:
            return {"status": "unsolvable", "log": say(["    Backtracker: stack empty -> unsolvable"])}
        frame = stack.pop()
        nodes = list(state["nodes"])
        nodes[frame["node"]] = {**nodes[frame["node"]], "status": "backtracked",
                                "reason": "subtree exhausted -- backtracked"}
        saved = backtracker.record(frame)
        cell = tuple(frame["cell"])
        line = (f"    Backtracker: undo {frame['value']}@{cell}; "
                f"{'dead end saved to long-term memory; ' if saved else ''}"
                f"siblings left: {frame['rest']}")
        return {"grid": frame["grid"], "stack": stack, "nodes": nodes,
                "current_node": frame["parent"],
                "pending": {"cell": frame["cell"], "candidates": frame["rest"],
                            "checked": True, "parent": frame["parent"]},
                "stats": bump(state, backtracks=1, dead_ends_saved=int(saved)),
                "log": say([line])}

    # ---------------- finish ----------------
    def finish(state):
        grid, status = state["grid"], state["status"]
        if status == "running":
            if find_empty(grid) is None:
                status = "solved" if not full_grid_violations(grid) else "invalid"
            else:
                status = "hop_limit"
        nodes = list(state["nodes"])
        if status in ("solved", "invalid"):
            for fr in state["stack"]:
                nodes[fr["node"]] = {**nodes[fr["node"]], "status": "solution"}
        return {"status": status, "nodes": nodes,
                "log": say([f"[finish] status={status} after {state['hops']} hops", grid_to_str(grid)])}

    g = StateGraph(SolveState)
    g.add_node("supervisor", supervisor)
    g.add_node("proposer", proposer_node)
    g.add_node("checker", checker_node)
    g.add_node("backtracker", backtracker_node)
    g.add_node("finish", finish)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"],
                            ["proposer", "checker", "backtracker", "finish"])
    for worker in ("proposer", "checker", "backtracker"):
        g.add_edge(worker, "supervisor")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


def run_config(thread_id: str, max_hops: int = MAX_HOPS) -> dict:
    # LangGraph's own recursion limit must sit above our MAX_HOPS so that
    # OUR hop limit is the one that fires (each hop = 2 graph steps).
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 2 * max_hops + 20}

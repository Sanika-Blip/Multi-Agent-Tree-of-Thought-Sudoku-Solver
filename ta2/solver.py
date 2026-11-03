"""
solver.py
---------
Wires memory + RAG + agents + graph into one call, and answers follow-up
questions about an earlier solve from the checkpointed thread state
(the U3-1 multi-turn demo).
"""

import time
import uuid

from langchain_ollama import ChatOllama

from agents import Proposer, RuleChecker, Backtracker
from graph import build_graph, initial_state, run_config, MAX_HOPS
from memory import get_checkpointer, get_store, get_profile, DEFAULT_USER
from rag import get_vectorstore, hint_for_cell
from sudoku import PUZZLES


def new_thread_id(prefix="solve"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def solve(puzzle_id, thread_id=None, *, max_hops=MAX_HOPS, temperature=0.4, guard=True,
          use_memory=True, fail_on_calls=(), notes="", resume=False, hint=None,
          verbose=True, tag="", user_id=DEFAULT_USER):
    thread_id = thread_id or new_thread_id(puzzle_id)
    store, saver = get_store(), get_checkpointer()
    if hint is None:
        hint = hint_for_cell(get_vectorstore(), k=1)
    proposer = Proposer(temperature=temperature, hint=hint, fail_on_calls=fail_on_calls)
    checker = RuleChecker(enabled=guard, store=store, user_id=user_id, use_memory=use_memory)
    # With the guard off the search can contain illegal boards, so it must
    # not write "dead ends" into long-term memory (that would poison it).
    backtracker = Backtracker(store=store, user_id=user_id, use_memory=use_memory and guard)
    graph = build_graph(proposer, checker, backtracker, checkpointer=saver,
                        verbose=verbose, tag=tag)
    if resume:
        inp = {"max_hops": max_hops, "status": "running"}
    else:
        order = get_profile(store, user_id).get("cell_order", "first")
        inp = initial_state(puzzle_id, PUZZLES[puzzle_id], notes=notes,
                            cell_order=order, max_hops=max_hops)
    t0 = time.time()
    final = graph.invoke(inp, run_config(thread_id, max_hops))
    final["wall_seconds"] = round(time.time() - t0, 2)
    final["thread_id"] = thread_id
    return final


def load_thread(thread_id):
    tup = get_checkpointer().get_tuple({"configurable": {"thread_id": thread_id}})
    return tup.checkpoint["channel_values"] if tup else None


FOLLOWUP_PROMPT = """You answer questions about an earlier Sudoku search, using ONLY the search record.
If the record is empty, reply exactly: I have no record of an earlier search on this thread.

Search record:
{record}

Question: {question}
Answer in at most 2 sentences:"""


def summarize(state) -> str:
    if not state:
        return "(empty)"
    bt = [ln.strip() for ln in state["log"] if "Backtracker:" in ln]
    pruned = [ln.strip() for ln in state["log"] if "PRUNED" in ln]
    return "\n".join([f"puzzle {state['puzzle_id']}, status {state['status']}, hops {state['hops']}",
                      f"stats {state['stats']}",
                      f"FIRST backtrack: {bt[0] if bt else '(none -- no backtracking was needed)'}",
                      "all backtracks in order:", *(bt or ["(none)"]),
                      "first pruned candidates:", *(pruned[:4] or ["(none)"])])


def ask(thread_id, question):
    record = summarize(load_thread(thread_id))
    reply = ChatOllama(model="qwen2.5:3b", temperature=0).invoke(
        FOLLOWUP_PROMPT.format(record=record, question=question)).content
    return reply.strip(), record

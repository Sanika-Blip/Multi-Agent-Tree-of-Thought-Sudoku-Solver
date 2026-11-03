"""
memory.py  (U3-1 short-term memory, U3-2 long-term memory)
---------
Short-term : LangGraph SqliteSaver checkpointer. Every solve runs on a
             thread_id; the full search state (board, DFS stack, search
             tree, log) is checkpointed after every hop, so a search can be
             resumed after a hop-limit stop or replayed, and follow-up
             questions on the same thread can see the earlier turn.
Long-term  : LangGraph SqliteStore (same SQLite file, survives a restart).
             Namespaces:
               (user_id, "profile")    -> preferences, e.g. cell order
               (user_id, "dead_ends")  -> board+cell+value placements that
                                          were proven to lead nowhere
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "memory.sqlite")
DEFAULT_USER = "demo_user"


def _connect(**kw) -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    # check_same_thread=False: the voting solvers run in worker threads.
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30, **kw)


def get_checkpointer() -> SqliteSaver:
    saver = SqliteSaver(_connect())
    saver.setup()
    return saver


def get_store() -> SqliteStore:
    # SqliteStore manages its own BEGIN/COMMIT, so it needs autocommit mode.
    store = SqliteStore(_connect(isolation_level=None))
    store.setup()
    return store


# ---- profile -----------------------------------------------------------

def get_profile(store, user_id: str = DEFAULT_USER) -> dict:
    item = store.get((user_id, "profile"), "prefs")
    return item.value if item else {}


def set_profile(store, user_id: str = DEFAULT_USER, **prefs) -> dict:
    profile = get_profile(store, user_id)
    profile.update(prefs)
    store.put((user_id, "profile"), "prefs", profile)
    return profile


# ---- dead-end patterns -------------------------------------------------
# Key = board before the placement + cell + value. Because the key is the
# exact board state, a stored dead end is a proven fact for that state and
# can never make the solver wrong; it can only save work.

def dead_end_key(board_key: str, cell, value: int) -> str:
    return f"{board_key}|{cell[0]}{cell[1]}|{value}"


def remember_dead_end(store, board_key: str, cell, value: int,
                      user_id: str = DEFAULT_USER) -> None:
    store.put((user_id, "dead_ends"), dead_end_key(board_key, cell, value),
              {"board": board_key, "cell": list(cell), "value": value})


def is_known_dead_end(store, board_key: str, cell, value: int,
                      user_id: str = DEFAULT_USER) -> bool:
    return store.get((user_id, "dead_ends"),
                     dead_end_key(board_key, cell, value)) is not None


def count_dead_ends(store, user_id: str = DEFAULT_USER) -> int:
    return len(store.search((user_id, "dead_ends"), limit=10_000))


def clear_dead_ends(store, user_id: str = DEFAULT_USER) -> int:
    items = store.search((user_id, "dead_ends"), limit=10_000)
    for it in items:
        store.delete((user_id, "dead_ends"), it.key)
    return len(items)

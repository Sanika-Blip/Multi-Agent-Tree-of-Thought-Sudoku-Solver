"""
seed_store.py
-------------
Creates/refreshes the persistent data used by TA2:
  - data/memory.sqlite : LangGraph Store with the (demo_user, "profile") entry,
                         dead-end memory cleared
  - data/chroma/       : Chroma index over corpus/ (nomic-embed-text)

Run once before the demos:  python seed_store.py
"""

from memory import get_store, set_profile, clear_dead_ends, DB_PATH
from rag import get_vectorstore, PERSIST_DIR


def main():
    store = get_store()
    profile = set_profile(store, name="Demo user", cell_order="first")
    removed = clear_dead_ends(store)
    print(f"[seed] SQLite store: {DB_PATH}")
    print(f"[seed]   profile = {profile}")
    print(f"[seed]   cleared {removed} old dead-end entries")
    vs = get_vectorstore(rebuild=True)
    print(f"[seed] Chroma: {PERSIST_DIR}  ({len(vs.get()['ids'])} chunks)")


if __name__ == "__main__":
    main()

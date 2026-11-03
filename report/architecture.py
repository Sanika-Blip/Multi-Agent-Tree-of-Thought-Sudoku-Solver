"""Draws report/architecture.png (TA2 architecture diagram)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

BOXES = {
    # name: (x, y, w, h, colour, label)
    "sup":   (4.0, 4.3, 2.6, 0.9, "#1565C0", "Supervisor\n(routing, MAX_HOPS)"),
    "prop":  (0.3, 2.4, 2.6, 0.9, "#6A1B9A", "Proposer\n(qwen2.5:3b)"),
    "check": (4.0, 2.4, 2.6, 0.9, "#2E7D32", "RuleChecker\n(guard agent)"),
    "back":  (7.7, 2.4, 2.6, 0.9, "#EF6C00", "Backtracker"),
    "fin":   (4.0, 6.0, 2.6, 0.7, "#37474F", "finish -> END"),
    "rag":   (0.3, 0.4, 2.6, 0.9, "#8E24AA", "Chroma RAG\nnomic-embed-text"),
    "store": (4.0, 0.4, 2.6, 0.9, "#00695C", "SqliteStore\nprofile, dead_ends"),
    "ckpt":  (7.7, 0.4, 2.6, 0.9, "#455A64", "SqliteSaver\ncheckpointer (thread_id)"),
    "vote":  (7.7, 5.9, 2.6, 0.9, "#AD1457", "Voter: 3 solvers\nin parallel"),
}
ARROWS = [("sup", "prop"), ("sup", "check"), ("sup", "back"), ("sup", "fin"),
          ("prop", "sup"), ("check", "sup"), ("back", "sup"),
          ("rag", "prop"), ("store", "check"), ("back", "store"),
          ("fin", "vote")]


def centre(name):
    x, y, w, h, *_ = BOXES[name]
    return x + w / 2, y + h / 2


def main():
    fig, ax = plt.subplots(figsize=(10.6, 7.2))
    for name, (x, y, w, h, col, label) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=col, ec="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color="white", fontsize=10, weight="bold")
    for a, b in ARROWS:
        (x1, y1), (x2, y2) = centre(a), centre(b)
        off = 0.25 if (b, a) in ARROWS else 0
        ax.annotate("", xy=(x2 + off, y2), xytext=(x1 + off, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#424242",
                                    shrinkA=28, shrinkB=28))
    ax.text(5.3, 3.7, "every worker returns to the Supervisor (1 hop each)",
            ha="center", fontsize=8.5, style="italic")
    ax.text(5.3, -0.25, "Short-term memory = whole DFS state checkpointed after every hop;  "
            "long-term memory = SQLite Store (survives restart)", ha="center", fontsize=8.5)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.5, 7.0)
    ax.axis("off")
    ax.set_title("TA2 architecture - multi-agent Tree-of-Thought Sudoku (LangGraph)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "architecture.png"), dpi=160)


if __name__ == "__main__":
    main()

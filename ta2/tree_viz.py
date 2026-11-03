"""
tree_viz.py
-----------
Step 4 of the assignment: render the full explored search tree -- including
pruned branches -- as a diagram. Pure matplotlib (no graphviz/dot binary
dependency), so it runs anywhere.

Color coding:
  green  = on the winning solution path
  grey   = valid placement that was later backtracked out of (dead end below it)
  red    = pruned immediately (violates a rule)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from search_tree import SearchNode


COLOR_SOLUTION = "#2E7D32"   # green
COLOR_BACKTRACK = "#9E9E9E"  # grey
COLOR_PRUNED = "#C62828"     # red
COLOR_ROOT = "#37474F"       # dark slate


def _layout(node: SearchNode, next_x, positions, depth=0):
    """Standard node-link tree layout: leaves get sequential x slots from a
    single global counter (so no two leaves anywhere in the tree ever
    collide), and every internal node sits at the average x of its
    children. `next_x` is a one-element list used as a mutable int."""
    if not node.children:
        x = next_x[0]
        next_x[0] += 1
    else:
        for child in node.children:
            _layout(child, next_x, positions, depth + 1)
        xs = [positions[ch.id][0] for ch in node.children]
        x = sum(xs) / len(xs)
    positions[node.id] = (x, -depth)


def render_tree(root: SearchNode, out_path: str, title: str = "Tree-of-Thought Search"):
    positions = {}
    _layout(root, [0], positions)

    all_nodes = []

    def collect(n):
        all_nodes.append(n)
        for ch in n.children:
            collect(ch)

    collect(root)

    max_depth = max(-positions[n.id][1] for n in all_nodes) if all_nodes else 1
    num_leaves = sum(1 for n in all_nodes if not n.children)
    fig_w = max(12, num_leaves * 1.15)
    fig_h = max(7, (max_depth + 1) * 1.3 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for n in all_nodes:
        if n.parent_id is None:
            continue
        parent = next((p for p in all_nodes if p.id == n.parent_id), None)
        if parent is None:
            continue
        x1, y1 = positions[parent.id]
        x2, y2 = positions[n.id]
        ax.plot([x1, x2], [y1, y2], color="#B0BEC5", linewidth=1.2, zorder=1)

    for n in all_nodes:
        x, y = positions[n.id]
        if n.status == "root":
            color = COLOR_ROOT
            label = "start"
        elif n.status == "pruned":
            color = COLOR_PRUNED
            label = f"{n.value}@{n.cell}"
        elif n.status == "backtracked":
            color = COLOR_BACKTRACK
            label = f"{n.value}@{n.cell}"
        elif n.status in ("solution", "valid"):
            color = COLOR_SOLUTION
            label = f"{n.value}@{n.cell}"
        else:
            color = "#455A64"
            label = str(n.value)

        box = FancyBboxPatch(
            (x - 0.32, y - 0.16), 0.64, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0, facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                 color="white", zorder=3, fontweight="bold")

    max_x = max((positions[n.id][0] for n in all_nodes), default=1)
    ax.set_xlim(-1, max_x + 1)
    legend_y_top = -max_depth - 0.9
    ax.set_ylim(legend_y_top - 1.0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    legend_items = [
        (COLOR_SOLUTION, "On winning path"),
        (COLOR_BACKTRACK, "Backtracked (dead end below)"),
        (COLOR_PRUNED, "Pruned (rule violation)"),
    ]
    for i, (color, text) in enumerate(legend_items):
        lx = 0 + i * (max_x / 2.6 if max_x > 0 else 3)
        ax.add_patch(FancyBboxPatch(
            (lx, legend_y_top - 0.15), 0.3, 0.22,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0, facecolor=color,
        ))
        ax.text(lx + 0.4, legend_y_top - 0.04, text, fontsize=9, va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    from llm_backend import MockBackend
    from tot_solver import run_tot

    result = run_tot(MockBackend())
    render_tree(result.root, "outputs/search_tree.png",
                title="ToT Search Tree - 4x4 Sudoku")
    print("Saved outputs/search_tree.png")

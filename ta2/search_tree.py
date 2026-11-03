"""
search_tree.py
--------------
The TA2 graph keeps its search tree as a flat, checkpoint-friendly list of
dicts. This module turns that list back into the SearchNode objects that
tree_viz.py (unchanged from TA1) knows how to draw.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchNode:
    id: int
    parent_id: Optional[int]
    cell: Optional[tuple]
    value: Optional[int]
    status: str
    reason: str
    depth: int
    children: List["SearchNode"] = field(default_factory=list)


def build_tree(flat_nodes: list) -> SearchNode:
    by_id = {}
    for n in flat_nodes:
        by_id[n["id"]] = SearchNode(
            id=n["id"], parent_id=n["parent"],
            cell=tuple(n["cell"]) if n["cell"] else None,
            value=n["value"], status=n["status"], reason=n["reason"], depth=0,
        )
    root = by_id[0]
    for n in flat_nodes:
        if n["parent"] is not None:
            by_id[n["parent"]].children.append(by_id[n["id"]])

    def set_depth(node, d):
        node.depth = d
        for ch in node.children:
            set_depth(ch, d + 1)

    set_depth(root, 0)
    return root


def save_tree(flat_nodes: list, png_path: str, json_path: str, title: str):
    from tree_viz import render_tree
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flat_nodes, f, indent=1)
    render_tree(build_tree(flat_nodes), png_path, title=title)

/* Browser port of the TA2 agent loop for the live demo.
   The Proposer is a fixed weak guesser (digits 1-4), so the page runs
   without Ollama. The RuleChecker, Backtracker, and Supervisor follow
   the same jobs as ta2/agents.py and ta2/graph.py. */
(function (root, factory) {
  var api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SudokuDemo = api;
})(typeof self !== "undefined" ? self : this, function () {
  var N = 4;
  var BOX = 2;

  var PUZZLES = {
    P1: [[2, 0, 0, 0], [0, 0, 2, 3], [0, 0, 0, 0], [1, 0, 0, 4]],
    P2: [[0, 2, 0, 3], [3, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0]],
    P3: [[0, 0, 0, 4], [3, 0, 0, 0], [4, 0, 0, 1], [1, 0, 0, 0]],
    P4: [[1, 4, 2, 0], [0, 0, 4, 0], [0, 2, 0, 0], [4, 0, 0, 0]],
    P5: [[0, 0, 3, 1], [0, 0, 0, 0], [4, 1, 2, 0], [0, 0, 0, 4]]
  };

  function clone(grid) {
    return grid.map(function (row) { return row.slice(); });
  }

  function invalidReason(grid, r, c, val) {
    var cc, rr, br, bc;
    for (cc = 0; cc < N; cc++) {
      if (cc !== c && grid[r][cc] === val) {
        return "row " + r + " already has " + val;
      }
    }
    for (rr = 0; rr < N; rr++) {
      if (rr !== r && grid[rr][c] === val) {
        return "column " + c + " already has " + val;
      }
    }
    br = Math.floor(r / BOX) * BOX;
    bc = Math.floor(c / BOX) * BOX;
    for (rr = br; rr < br + BOX; rr++) {
      for (cc = bc; cc < bc + BOX; cc++) {
        if ((rr !== r || cc !== c) && grid[rr][cc] === val) {
          return "box already has " + val + " at (" + rr + "," + cc + ")";
        }
      }
    }
    return "";
  }

  function findEmpty(grid) {
    var r, c;
    for (r = 0; r < N; r++) {
      for (c = 0; c < N; c++) {
        if (grid[r][c] === 0) return [r, c];
      }
    }
    return null;
  }

  function violations(grid) {
    var problems = [];
    var r, c, br, bc, vals, i;
    for (r = 0; r < N; r++) {
      vals = [];
      for (c = 0; c < N; c++) vals.push(grid[r][c]);
      if (vals.slice().sort().join(",") !== "1,2,3,4") {
        problems.push("row " + r + " is [" + vals.join(", ") + "]");
      }
    }
    for (c = 0; c < N; c++) {
      vals = [];
      for (r = 0; r < N; r++) vals.push(grid[r][c]);
      if (vals.slice().sort().join(",") !== "1,2,3,4") {
        problems.push("column " + c + " is [" + vals.join(", ") + "]");
      }
    }
    for (br = 0; br < N; br += BOX) {
      for (bc = 0; bc < N; bc += BOX) {
        vals = [];
        for (r = br; r < br + BOX; r++) {
          for (c = bc; c < bc + BOX; c++) vals.push(grid[r][c]);
        }
        if (vals.slice().sort().join(",") !== "1,2,3,4") {
          problems.push("box (" + br + "," + bc + ") is [" + vals.join(", ") + "]");
        }
      }
    }
    for (r = 0; r < N; r++) {
      for (c = 0; c < N; c++) {
        if (grid[r][c] === 0) problems.push("cell (" + r + "," + c + ") is empty");
      }
    }
    return problems;
  }

  function runSearch(puzzle, options) {
    var guard = !options || options.guard !== false;
    var maxHops = (options && options.maxHops) || (guard ? 220 : 80);
    var grid = clone(puzzle);
    var steps = [];
    var stack = [];
    var hops = 0;
    var pruned = 0;
    var backtracks = 0;
    var status = "running";

    function push(partial) {
      steps.push({
        hop: hops,
        pruned: pruned,
        backtracks: backtracks,
        grid: clone(grid),
        agent: partial.agent,
        kind: partial.kind,
        cell: partial.cell || null,
        value: partial.value || null,
        text: partial.text
      });
    }

    function backtrack() {
      backtracks += 1;
      while (stack.length) {
        var frame = stack.pop();
        grid[frame.r][frame.c] = 0;
        push({
          agent: "Backtracker",
          kind: "undo",
          cell: [frame.r, frame.c],
          text: "Dead end. Undid (" + frame.r + ", " + frame.c + ") and kept the failure in memory."
        });
        if (frame.rest.length) {
          var next = frame.rest.shift();
          grid[frame.r][frame.c] = next;
          stack.push(frame);
          push({
            agent: "Supervisor",
            kind: "place",
            cell: [frame.r, frame.c],
            value: next,
            text: "Resumed with the next saved digit " + next + " at (" + frame.r + ", " + frame.c + ")."
          });
          return true;
        }
      }
      push({
        agent: "Supervisor",
        kind: "stuck",
        text: "Every branch is exhausted."
      });
      return false;
    }

    while (hops < maxHops && status === "running") {
      hops += 1;
      var pos = findEmpty(grid);
      push({
        agent: "Supervisor",
        kind: "route",
        cell: pos,
        text: pos
          ? "Hop " + hops + ". The next empty cell is (" + pos[0] + ", " + pos[1] + "). Sending it to the Proposer."
          : "Hop " + hops + ". The board is full."
      });

      if (!pos) {
        var problems = violations(grid);
        status = problems.length ? "invalid" : "solved";
        push({
          agent: "Supervisor",
          kind: status === "solved" ? "solved" : "invalid",
          text: status === "solved"
            ? "Solved. Every row, column, and 2×2 box contains 1–4 exactly once."
            : "The board is full but illegal. " + problems.slice(0, 3).join("; ") + "."
        });
        break;
      }

      var proposed = [1, 2, 3, 4];
      push({
        agent: "Proposer",
        kind: "propose",
        cell: pos,
        text: "Suggests " + proposed.join(", ") + " for (" + pos[0] + ", " + pos[1] + "). This live proposer always lists 1–4, the way a weak model dumps digits for the guard to screen."
      });

      var accepted = [];
      var notes = [];
      var v, reason;
      for (v = 0; v < proposed.length; v++) {
        reason = invalidReason(grid, pos[0], pos[1], proposed[v]);
        if (guard && reason) {
          pruned += 1;
          notes.push(proposed[v] + " pruned — " + reason);
        } else if (!guard && reason) {
          accepted.push(proposed[v]);
          notes.push(proposed[v] + " kept — " + reason + " (guard off)");
        } else {
          accepted.push(proposed[v]);
        }
      }

      push({
        agent: "RuleChecker",
        kind: notes.length ? "prune" : "accept",
        cell: pos,
        text: notes.length
          ? notes.join(" · ")
          : "All four digits are legal here. None were pruned."
      });

      if (!accepted.length) {
        push({
          agent: "Backtracker",
          kind: "dead",
          cell: pos,
          text: "No legal digit remains for (" + pos[0] + ", " + pos[1] + ")."
        });
        if (!backtrack()) status = "stuck";
        continue;
      }

      var first = accepted[0];
      var rest = accepted.slice(1);
      grid[pos[0]][pos[1]] = first;
      stack.push({ r: pos[0], c: pos[1], rest: rest });
      reason = invalidReason(grid, pos[0], pos[1], first);
      push({
        agent: "Supervisor",
        kind: reason ? "illegal" : "place",
        cell: pos,
        value: first,
        text: reason
          ? "Guard is off, so " + first + " was placed at (" + pos[0] + ", " + pos[1] + ") even though it breaks a rule."
          : "Placed " + first + " at (" + pos[0] + ", " + pos[1] + ")." +
            (rest.length ? " Saved siblings: " + rest.join(", ") + "." : "")
      });
    }

    if (status === "running" && hops >= maxHops) {
      status = "limited";
      push({
        agent: "Supervisor",
        kind: "limit",
        text: "Stopped at the hop limit (" + maxHops + "), the same kind of cap the Python supervisor uses."
      });
    }

    return { steps: steps, status: status, hops: hops, pruned: pruned, backtracks: backtracks };
  }

  return {
    PUZZLES: PUZZLES,
    runSearch: runSearch,
    violations: violations,
    invalidReason: invalidReason
  };
});

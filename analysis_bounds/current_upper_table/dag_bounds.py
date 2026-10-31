"""Certified relaxations for scene B makespan, without enumerating schedules.

No original input or evaluator is changed.  Original COPY nodes are discarded
and one compulsory DDR read/write is rebuilt per boundary tensor.  All core
assignment, finite-memory, communication-delay, FIFO and extra-copy constraints
are relaxed.  Compute pipes retain their names (including PIPE_FIX).

The DDR resource is a fluid server of capacity one in exclusive-copy-cycle
units.  A resource cut uses only integrated service, not serial DDR execution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from fractions import Fraction
import json
from pathlib import Path


COPY_TYPES = {"COPY_IN", "COPY_OUT"}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_relaxation(graph: dict, bandwidth=60) -> dict:
    """Build an optimistic DAG of compute and compulsory boundary transfers.

    For a shared input choose the first completed actual read.  It precedes all
    consumers.  For an output with multiple producers choose the output copy
    belonging to a core with a latest-finishing producer.  This copy starts
    after every producer has finished.  These choices prove that keeping just
    one transfer per tensor is a relaxation even with replication.
    """
    bw = Fraction(str(bandwidth))
    if bw <= 0:
        raise ValueError("bandwidth must be positive")
    original_ops = {op["id"]: op for op in graph["ops"]}
    tensors = {t["id"]: t for t in graph["tensors"]}
    compute = {i: op for i, op in original_ops.items()
               if op["op"] not in COPY_TYPES}
    producers, consumers, original_consumers = (defaultdict(set) for _ in range(3))
    direct = []
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if a in compute and b in tensors:
            producers[b].add(a)
        if a in tensors and b in compute:
            consumers[a].add(b)
        if a in tensors and b in original_ops:
            original_consumers[a].add(b)
        if a in compute and b in compute:
            direct.append((a, b))
    nodes = []
    for i, op in sorted(compute.items()):
        if isinstance(op["cycles"], bool) or op["cycles"] < 0:
            raise ValueError(f"invalid compute cycles for op {i}")
        nodes.append({"key": f"op:{i}", "original_id": i,
                      "kind": "compute", "resource": op["pipe"],
                      "duration": max(1, op["cycles"])})
    edges = [(f"op:{a}", f"op:{b}") for a, b in direct]
    for tid, tensor in sorted(tensors.items()):
        ps, cs = producers[tid], consumers[tid]
        edges.extend((f"op:{a}", f"op:{b}") for a in ps for b in cs)
        service = max(1, ceil_div(tensor["size"] * bw.denominator, bw.numerator))
        if cs and not ps:
            key = f"input:{tid}"
            nodes.append({"key": key, "original_id": tid, "kind": "input",
                          "resource": "DDR", "duration": service})
            edges.extend((key, f"op:{c}") for c in cs)
        has_copy_out = any(original_ops[c]["op"] == "COPY_OUT"
                           for c in original_consumers[tid])
        if ps and (has_copy_out or not cs):
            key = f"output:{tid}"
            nodes.append({"key": key, "original_id": tid, "kind": "output",
                          "resource": "DDR", "duration": service})
            edges.extend((f"op:{p}", key) for p in ps)
    index = {node["key"]: i for i, node in enumerate(nodes)}
    preds = [set() for _ in nodes]
    succs = [set() for _ in nodes]
    for a, b in edges:
        u, v = index[a], index[b]
        preds[v].add(u)
        succs[u].add(v)
    degree = list(map(len, preds))
    ready = deque(i for i, d in enumerate(degree) if not d)
    topo = []
    while ready:
        u = ready.popleft()
        topo.append(u)
        for v in sorted(succs[u]):
            degree[v] -= 1
            if not degree[v]:
                ready.append(v)
    if len(topo) != len(nodes):
        raise ValueError("relaxed graph contains a cycle")
    return {"nodes": nodes, "preds": preds, "succs": succs, "topo": topo}


class PrefixAddMax:
    """Prefix range additions and prefix maximum queries, with an argmax."""

    def __init__(self, values):
        self.n = len(values)
        self.values = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.index = [0] * (4 * self.n)
        self._build(1, 0, self.n - 1, values)

    def _build(self, t, lo, hi, values):
        if lo == hi:
            self.values[t], self.index[t] = values[lo], lo
            return
        mid = (lo + hi) // 2
        self._build(2*t, lo, mid, values)
        self._build(2*t+1, mid+1, hi, values)
        self._pull(t)

    def _pull(self, t):
        child = 2*t if self.values[2*t] >= self.values[2*t+1] else 2*t+1
        self.values[t], self.index[t] = self.values[child], self.index[child]

    def _push(self, t):
        if self.lazy[t]:
            for child in (2*t, 2*t+1):
                self.values[child] += self.lazy[t]
                self.lazy[child] += self.lazy[t]
            self.lazy[t] = 0

    def add(self, end, amount):
        self._add(1, 0, self.n-1, end, amount)

    def _add(self, t, lo, hi, end, amount):
        if hi <= end:
            self.values[t] += amount
            self.lazy[t] += amount
            return
        self._push(t)
        mid = (lo + hi) // 2
        self._add(2*t, lo, mid, end, amount)
        if end > mid:
            self._add(2*t+1, mid+1, hi, end, amount)
        self._pull(t)

    def maximum(self, end):
        return self._maximum(1, 0, self.n-1, end)

    def _maximum(self, t, lo, hi, end):
        if hi <= end:
            return self.values[t], self.index[t]
        self._push(t)
        mid = (lo + hi) // 2
        answer = self._maximum(2*t, lo, mid, end)
        if end > mid:
            other = self._maximum(2*t+1, mid+1, hi, end)
            if other[0] > answer[0]:
                answer = other
        return answer


def strongest_resource_cut(jobs, capacity):
    """Exact max of r0 + ceil(sum(p for r>=r0,q>=q0)/m) + q0.

    jobs contain (release lower bound, tail lower bound, duration, node index).
    A descending release sweep and prefix-update segment tree make the search
    O(n log n), instead of enumerating O(n^2) threshold pairs.  Empty subsets
    are excluded explicitly.
    """
    if not jobs:
        return {"lower_bound": 0}
    qs = sorted({q for r, q, p, i in jobs})
    qi = {q: i for i, q in enumerate(qs)}
    grouped = defaultdict(list)
    for r, q, p, i in jobs:
        grouped[r].append((q, p, i))
    tree = PrefixAddMax([capacity*q for q in qs])
    best = {"lower_bound": 0}
    active_q = -1
    for r in sorted(grouped, reverse=True):
        for q, p, i in grouped[r]:
            tree.add(qi[q], p)
            active_q = max(active_q, qi[q])
        value, index = tree.maximum(active_q)
        lb = r + ceil_div(value, capacity)
        if lb > best["lower_bound"]:
            q = qs[index]
            best = {"lower_bound": lb, "release_threshold": r,
                    "tail_threshold": q, "capacity": capacity,
                    "work": value-capacity*q}
    selected = [j for j in jobs if j[0] >= best["release_threshold"]
                and j[1] >= best["tail_threshold"]]
    best["node_count"] = len(selected)
    best["example_nodes"] = [j[3] for j in selected[:8]]
    # A valid additional nonpreemptive packing cut for the selected window.
    # Equal-duration compute jobs need ceil(n/m) whole slots on some machine.
    # This does not apply to the fluid DDR resource.
    return best


def reachability_work(model):
    """Per-node ancestor/descendant workload, using integer reachability bitsets."""
    nodes, topo = model["nodes"], model["topo"]
    pred, succ = model["preds"], model["succs"]
    ancestors, descendants = [0]*len(nodes), [0]*len(nodes)
    for v in topo:
        mask = 0
        for u in pred[v]:
            mask |= ancestors[u] | (1 << u)
        ancestors[v] = mask
    for v in reversed(topo):
        mask = 0
        for u in succ[v]:
            mask |= descendants[u] | (1 << u)
        descendants[v] = mask
    grouped_masks = defaultdict(int)
    for i, node in enumerate(nodes):
        grouped_masks[node["resource"], node["duration"]] |= 1 << i
    resources = sorted({n["resource"] for n in nodes})
    before = {r: [0]*len(nodes) for r in resources}
    after = {r: [0]*len(nodes) for r in resources}
    for (resource, duration), mask in grouped_masks.items():
        for i in range(len(nodes)):
            before[resource][i] += (ancestors[i] & mask).bit_count()*duration
            after[resource][i] += (descendants[i] & mask).bit_count()*duration
    return before, after


def compute_bounds(graph, bandwidth=60, core_counts=(1, 2, 3, 4, 5),
                   ancestor_work=True):
    model = build_relaxation(graph, bandwidth)
    nodes, topo = model["nodes"], model["topo"]
    durations = [n["duration"] for n in nodes]
    resources = sorted({n["resource"] for n in nodes})
    resource_nodes = {r: [i for i, n in enumerate(nodes) if n["resource"] == r]
                      for r in resources}
    if ancestor_work:
        before, after = reachability_work(model)
    else:
        before, after = {}, {}
    basic_r = [0]*len(nodes)
    basic_q = [0]*len(nodes)
    for v in topo:
        basic_r[v] = max((basic_r[u]+durations[u] for u in model["preds"][v]), default=0)
    for v in reversed(topo):
        basic_q[v] = max((basic_q[u]+durations[u] for u in model["succs"][v]), default=0)
    cp = max((basic_r[i]+durations[i] for i in range(len(nodes))), default=0)
    result = {"relaxation_nodes": len(nodes),
              "compulsory_input_count": sum(n["kind"] == "input" for n in nodes),
              "compulsory_output_count": sum(n["kind"] == "output" for n in nodes),
              "resource_work": {r: sum(durations[i] for i in ids)
                                for r, ids in resource_nodes.items()},
              "critical_path": cp, "cores": {}}
    for k in core_counts:
        if k < 1:
            raise ValueError("core count must be positive")
        capacities = {r: 1 if r == "DDR" else k for r in resources}
        r, q = basic_r[:], basic_q[:]
        if ancestor_work:
            for i in range(len(nodes)):
                r[i] = max(r[i], max((ceil_div(before[res][i], capacities[res])
                                     for res in resources), default=0))
                q[i] = max(q[i], max((ceil_div(after[res][i], capacities[res])
                                     for res in resources), default=0))
            for v in topo:
                r[v] = max(r[v], max((r[u]+durations[u] for u in model["preds"][v]), default=0))
            for v in reversed(topo):
                q[v] = max(q[v], max((q[u]+durations[u] for u in model["succs"][v]), default=0))
        cuts = {}
        for res, ids in resource_nodes.items():
            jobs = [(r[i], q[i], durations[i], i) for i in ids]
            cuts[res] = strongest_resource_cut(jobs, capacities[res])
            if res != "DDR":
                cut = cuts[res]
                selected = [j for j in jobs if j[0] >= cut["release_threshold"]
                            and j[1] >= cut["tail_threshold"]]
                sizes = {j[2] for j in selected}
                if len(sizes) == 1:
                    size = sizes.pop()
                    packed = (cut["release_threshold"] + cut["tail_threshold"]
                              + ceil_div(len(selected), capacities[res])*size)
                    if packed > cut["lower_bound"]:
                        cut["preemptive_workload_lower_bound"] = cut["lower_bound"]
                        cut["lower_bound"] = packed
                        cut["indivisible_equal_duration"] = size
            cuts[res]["example_nodes"] = [nodes[i]["key"] for i in cuts[res]["example_nodes"]]
        via_node = max((r[i]+durations[i]+q[i] for i in range(len(nodes))), default=0)
        all_values = {res: cut["lower_bound"] for res, cut in cuts.items()}
        all_values["precedence_and_ancestor_work"] = via_node
        winner = max(all_values, key=all_values.get) if all_values else "empty"
        result["cores"][str(k)] = {
            "lower_bound": max(all_values.values(), default=0),
            "dominant_bound": winner,
            "precedence_and_ancestor_work": via_node,
            "resource_cuts": cuts,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="*", type=Path)
    parser.add_argument("--bandwidth", type=str, default="60")
    parser.add_argument("--output", type=Path,
                        default=Path("analysis_bounds/dag_bounds.json"))
    parser.add_argument("--no-ancestor-work", action="store_true")
    args = parser.parse_args()
    paths = args.graphs or sorted(Path("data").glob("case_*.json"))
    answer = {"bandwidth_bytes_per_cycle": str(args.bandwidth), "cases": {}}
    for path in paths:
        row = compute_bounds(json.loads(path.read_text()), args.bandwidth,
                             ancestor_work=not args.no_ancestor_work)
        answer["cases"][path.stem] = row
        print(path.stem, {k: x["lower_bound"] for k, x in row["cores"].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(answer, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()

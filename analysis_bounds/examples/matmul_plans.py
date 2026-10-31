"""Small deterministic whole-accumulation-chain candidate family for cases 00–06.

Only structural inference and plan construction happen here. No evaluator is
called, no input is changed, and single-op subgraphs expose Step1 ordering.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path

BLOCK_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2),
                (1, 3), (3, 1), (2, 3), (3, 2),
                (3, 3), (3, 4), (4, 3), (4, 4), (3, 6), (6, 3))


@dataclass
class MatmulStructure:
    rows: int
    columns: int
    depth: int
    chains: dict
    row_inputs: dict
    column_inputs: dict
    tensor_by_id: dict
    outputs: dict
    compute_ids: set

    def summary(self):
        row_sizes = [self.tensor_by_id[t]["size"] for t in self.row_inputs.values()]
        col_sizes = [self.tensor_by_id[t]["size"] for t in self.column_inputs.values()]
        partial = [self.tensor_by_id[t]["size"]
                   for chain in self.chains.values() for op in chain[:-1]
                   for t in self.outputs[op]]
        final = [self.tensor_by_id[t]["size"]
                 for chain in self.chains.values() for t in self.outputs[chain[-1]]]
        return {
            "rows": self.rows, "columns": self.columns, "depth": self.depth,
            "compute_ops": len(self.compute_ids),
            "row_input_size_range": [min(row_sizes), max(row_sizes)],
            "column_input_size_range": [min(col_sizes), max(col_sizes)],
            "partial_output_size_range": [min(partial), max(partial)] if partial else [],
            "final_output_size_range": [min(final), max(final)],
            "block_working_set_estimates": {
                f"{r}x{c}": self.working_set_estimate(r, c)
                for r, c in BLOCK_SHAPES},
            "working_set_note": (
                "Estimate counts one row/column operand per depth, all block partials, "
                "and one new output. It is not a peak-memory certificate: retained "
                "inputs, prefetching, pipeline overlap and Step2 spills are excluded."),
        }

    def working_set_estimate(self, block_rows, block_columns):
        r, c = min(block_rows, self.rows), min(block_columns, self.columns)
        a = max(self.tensor_by_id[t]["size"] for t in self.row_inputs.values())
        b = max(self.tensor_by_id[t]["size"] for t in self.column_inputs.values())
        partial = max((sum(self.tensor_by_id[t]["size"] for t in self.outputs[op])
                       for chain in self.chains.values() for op in chain[:-1]), default=0)
        output = max(sum(self.tensor_by_id[t]["size"] for t in self.outputs[op])
                     for chain in self.chains.values() for op in chain)
        return r * a + c * b + r * c * partial + output


def infer_structure(graph):
    """Infer M/N from the bipartite shared-input graph at reduction depth zero."""
    ops = {o["id"]: o for o in graph["ops"]
           if o["op"] not in ("COPY_IN", "COPY_OUT")}
    if not ops or any(o["op"] not in ("A_MUL_B", "A_MULACC_B") for o in ops.values()):
        raise ValueError("Expected pure A_MUL_B/A_MULACC_B graph")
    tensors = {t["id"]: t for t in graph["tensors"]}
    inputs, outputs = defaultdict(list), defaultdict(list)
    producers, consumers = defaultdict(set), defaultdict(set)
    for e in graph["edges"]:
        a, b = e["source"], e["target"]
        if a in ops and b in tensors:
            outputs[a].append(b)
            producers[b].add(a)
        elif a in tensors and b in ops:
            inputs[b].append(a)
            consumers[a].add(b)
        elif a in ops and b in ops:
            raise ValueError("Unexpected direct compute edge")
    preds, succs = defaultdict(set), defaultdict(set)
    external = {}
    for op in ops:
        external[op] = [t for t in inputs[op] if not producers[t]]
        if len(external[op]) != 2:
            raise ValueError(f"Op {op} needs exactly two external operands")
        for t in inputs[op]:
            for parent in producers[t]:
                preds[op].add(parent)
                succs[parent].add(op)
    if any(len(preds[op]) > 1 or len(succs[op]) > 1 for op in ops):
        raise ValueError("Compute graph is not a disjoint union of chains")
    chains = []
    visited = set()
    for source in sorted(op for op in ops if not preds[op]):
        chain = []
        op = source
        while True:
            if op in visited:
                raise ValueError("Repeated operation in chain traversal")
            visited.add(op)
            chain.append(op)
            if not succs[op]:
                break
            op = next(iter(succs[op]))
        chains.append(chain)
    if visited != set(ops) or len({len(chain) for chain in chains}) != 1:
        raise ValueError("Incomplete or unequal-length accumulation chains")

    adjacency = defaultdict(set)
    for chain in chains:
        a, b = external[chain[0]]
        adjacency[a].add(b)
        adjacency[b].add(a)
    colors = {min(adjacency): 0}
    queue = deque(colors)
    while queue:
        v = queue.popleft()
        for nxt in adjacency[v]:
            if nxt in colors and colors[nxt] == colors[v]:
                raise ValueError("Operand sharing is not bipartite")
            if nxt not in colors:
                colors[nxt] = 1 - colors[v]
                queue.append(nxt)
    if len(colors) != len(adjacency):
        raise ValueError("Expected one connected operand-sharing graph")
    row_keys = sorted(t for t, color in colors.items() if color == 0)
    col_keys = sorted(t for t, color in colors.items() if color == 1)
    row_index = {t: i for i, t in enumerate(row_keys)}
    col_index = {t: i for i, t in enumerate(col_keys)}
    indexed = {}
    for chain in chains:
        a, b = external[chain[0]]
        if colors[a] != 0:
            a, b = b, a
        cell = (row_index[a], col_index[b])
        if cell in indexed:
            raise ValueError("Multiple chains for the same output cell")
        indexed[cell] = chain
    if len(indexed) != len(row_keys) * len(col_keys):
        raise ValueError("Output grid is not a complete rectangle")

    chain_by_op = {op: cell for cell, chain in indexed.items() for op in chain}
    tensor_chain_users = {t: {chain_by_op[op] for op in users}
                          for t, users in consumers.items() if not producers[t]}
    row_users = {r: {(r, c) for c in range(len(col_keys))} for r in range(len(row_keys))}
    col_users = {c: {(r, c) for r in range(len(row_keys))} for c in range(len(col_keys))}
    row_inputs, col_inputs = {}, {}
    for (r, c), chain in indexed.items():
        for depth, op in enumerate(chain):
            row_candidates = [t for t in external[op] if tensor_chain_users[t] == row_users[r]]
            col_candidates = [t for t in external[op] if tensor_chain_users[t] == col_users[c]]
            if len(row_candidates) != 1 or len(col_candidates) != 1:
                raise ValueError("Operand sharing differs from the inferred grid")
            for dest, key, value in ((row_inputs, (r, depth), row_candidates[0]),
                                     (col_inputs, (c, depth), col_candidates[0])):
                if key in dest and dest[key] != value:
                    raise ValueError("Inconsistent input tile at reduction depth")
                dest[key] = value
    return MatmulStructure(len(row_keys), len(col_keys), len(chains[0]), indexed,
                           row_inputs, col_inputs, tensors, dict(outputs), set(ops))


def _blocks(r0, r1, c0, c1, br, bc):
    for r in range(r0, r1, br):
        for c in range(c0, c1, bc):
            yield [(i, j) for i in range(r, min(r + br, r1))
                   for j in range(c, min(c + bc, c1))]


def validate_structure_plan(structure, plan):
    """Coverage, one-core-per-chain and per-chain topology; no evaluator call."""
    schedules = plan["core_schedules"]
    flat = [op for order in schedules for op in order]
    if len(flat) != len(set(flat)) or set(flat) != structure.compute_ids:
        raise ValueError("Plan does not cover each compute op exactly once")
    if plan["node_to_subgraph"] != {str(op): op for op in structure.compute_ids}:
        raise ValueError("Expected one singleton subgraph per operation")
    positions = {op: (core, i) for core, order in enumerate(schedules)
                 for i, op in enumerate(order)}
    for chain in structure.chains.values():
        if len({positions[op][0] for op in chain}) != 1:
            raise ValueError("An accumulation chain was split across cores")
        if any(positions[a][1] >= positions[b][1] for a, b in zip(chain, chain[1:])):
            raise ValueError("Per-chain topological order was reversed")


def generate_candidates(graph, num_cores, *, block_shapes=BLOCK_SHAPES,
                        max_working_set_bytes=None):
    """Yield (name, plan), up to 42 candidates before structural deduplication.

    max_working_set_bytes filters by the documented estimate only. It does not
    claim that evaluator schedules avoid spills or obey a certified peak.
    """
    if num_cores < 1:
        raise ValueError("num_cores must be positive")
    shape = infer_structure(graph)
    seen = set()
    for br, bc in block_shapes:
        if br < 1 or bc < 1:
            raise ValueError("Block dimensions must be positive")
        if (max_working_set_bytes is not None and
                shape.working_set_estimate(br, bc) > max_working_set_bytes):
            continue
        for partition in ("row_bands", "column_bands", "tiles_round_robin"):
            per_core_blocks = [[] for _ in range(num_cores)]
            if partition == "tiles_round_robin":
                for index, block in enumerate(_blocks(0, shape.rows, 0, shape.columns, br, bc)):
                    per_core_blocks[index % num_cores].append(block)
            else:
                dimension = shape.rows if partition == "row_bands" else shape.columns
                for core in range(num_cores):
                    lo, hi = core * dimension // num_cores, (core + 1) * dimension // num_cores
                    limits = ((lo, hi, 0, shape.columns) if partition == "row_bands"
                              else (0, shape.rows, lo, hi))
                    per_core_blocks[core] = list(_blocks(*limits, br, bc))
            orders = [[shape.chains[cell][depth]
                       for block in blocks for depth in range(shape.depth) for cell in block]
                      for blocks in per_core_blocks]
            signature = tuple(tuple(order) for order in orders)
            if signature in seen:
                continue
            seen.add(signature)
            plan = {"node_to_subgraph": {str(op): op for op in sorted(shape.compute_ids)},
                    "core_schedules": orders}
            validate_structure_plan(shape, plan)
            yield f"{partition}_block_{br}x{bc}", plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="+")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-working-set-bytes", type=int)
    args = parser.parse_args(argv)
    for filename in args.graphs:
        graph = json.loads(Path(filename).read_text())
        structure = infer_structure(graph)
        records = []
        for name, plan in generate_candidates(
                graph, args.cores, max_working_set_bytes=args.max_working_set_bytes):
            record = {"name": name, "compute_ops_per_core": list(map(len, plan["core_schedules"]))}
            if args.output_dir:
                directory = args.output_dir / Path(filename).stem / f"cores_{args.cores}"
                directory.mkdir(parents=True, exist_ok=True)
                output = directory / f"{name}.json"
                output.write_text(json.dumps(plan, separators=(",", ":")) + "\n")
                record["plan_file"] = str(output.resolve())
            records.append(record)
        print(json.dumps({"case": Path(filename).stem, **structure.summary(),
                          "candidate_count": len(records), "candidates": records}), flush=True)


if __name__ == "__main__":
    main()

"""Certified finite-memory I/O lower bounds for the regular matmul cases.

This file never changes the input graph or fixed evaluator.  It recognizes the
actual matrix-index incidence pattern before applying a three-projection bound.
It allows arbitrary operation order, splitting chains between cores, and spills.

Proof (all sizes below are integer multiples of ``unit_bytes``):
* A phase containing n distinct products has projections A={(i,k)}, B={(k,j)},
  C={(i,j)} of cardinalities a,b,c.  The discrete projection inequality gives
  n^2 <= a*b*c.  Also n <= J*a, I*b, K*c.
* Its A and B projections require incoming values. Each represented C chain
  requires an incoming and outgoing accumulator, each of size wC, except that
  a chain's first product needs no incoming accumulator, and its final output
  has size wF. Charge virtual I/O wC to each first product and wC-wF to each
  final product. Then weighted boundary volume is at least
      wA*a + wB*b + 2*wC*c.
  Initial/final resident values supply at most 2*M. All other boundary values
  require real or virtual I/O in the phase. Repeated crossings only add I/O.
* Serialize each core's completion events. This is a relaxation: every operand
  remains available until its consumer completes in the actual evaluator, and
  outputs were already allocated. Thus snapshots fit its private L1 capacity.
  An op and its virtual charge are one event. Greedily end a phase when charges
  first reach q. A phase charge is <= q+dmax-1. If total augmented I/O is A,
  the number of phases across k cores is <= floor(A/q)+k.
* Enumerating projection counts, not schedules, yields an integer H such that
  every phase executes <= H products. Thus
      Q >= q * max(0, ceil(N/H)-k) - virtual_total.
  Optimize over several hundred q values and divide the certified byte count
  by global DDR bandwidth. COPY rounding can only make actual execution longer.

The result is intentionally a lower bound; it does not assume whole-chain
co-location, a particular blocking scheme, or a fixed Step1 order.
"""

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from math import ceil, gcd, isqrt


def _regular_matmul(graph):
    """Recover and verify a complete I x J x K product incidence pattern."""
    all_ops = {op['id']: op for op in graph['ops']}
    ops = {oid: op for oid, op in all_ops.items()
           if op['op'] not in ('COPY_IN', 'COPY_OUT')}
    if not ops or any(op['op'] not in ('A_MUL_B', 'A_MULACC_B')
                      for op in ops.values()):
        raise ValueError('not a regular matmul graph')
    tensors = {t['id']: t for t in graph['tensors']}
    incoming, outgoing = defaultdict(list), defaultdict(list)
    producers, consumers = defaultdict(list), defaultdict(list)
    for edge in graph['edges']:
        src, dst = edge['source'], edge['target']
        if src in all_ops and dst in tensors:
            outgoing[src].append(dst)
            if src in ops:
                producers[dst].append(src)
        elif src in tensors and dst in all_ops:
            incoming[dst].append(src)
            if dst in ops:
                consumers[src].append(dst)
        else:
            raise ValueError('matrix recognizer requires op-tensor edges')
    if any(len(v) != 1 for v in producers.values()):
        raise ValueError('multiple producers')
    op_pred, op_succ, operands = {}, {}, {}
    for oid, op in ops.items():
        if len(outgoing[oid]) != 1:
            raise ValueError('matmul needs exactly one output per op')
        tid = outgoing[oid][0]
        if len(consumers[tid]) > 1:
            raise ValueError('branched accumulator chain')
        op_succ[oid] = consumers[tid][0] if consumers[tid] else None
        source_tensors = [t for t in incoming[oid] if not producers[t]]
        accumulator_tensors = [t for t in incoming[oid] if producers[t]]
        if len(source_tensors) != 2 or len(accumulator_tensors) > 1:
            raise ValueError('not a two-operand accumulator operation')
        operands[oid] = tuple(source_tensors)
        op_pred[oid] = (producers[accumulator_tensors[0]][0]
                        if accumulator_tensors else None)
        if bool(accumulator_tensors) != (op['op'] == 'A_MULACC_B'):
            raise ValueError('inconsistent accumulator root')

    chains, seen = [], set()
    for root in (oid for oid in ops if op_pred[oid] is None):
        chain, oid = [], root
        while oid is not None:
            if oid in seen:
                raise ValueError('repeated/cyclic accumulator chain')
            seen.add(oid)
            chain.append(oid)
            oid = op_succ[oid]
        chains.append(chain)
    if seen != set(ops) or not chains:
        raise ValueError('incomplete chain decomposition')
    K = len(chains[0])
    if any(len(c) != K for c in chains):
        raise ValueError('nonuniform reduction lengths')
    row_keys, col_keys, pairs = {}, {}, set()
    a_ids, b_ids = set(), set()
    intermediate_sizes, final_sizes = set(), set()
    for chain in chains:
        aa = tuple(operands[o][0] for o in chain)
        bb = tuple(operands[o][1] for o in chain)
        i = row_keys.setdefault(aa, len(row_keys))
        j = col_keys.setdefault(bb, len(col_keys))
        if (i, j) in pairs:
            raise ValueError('duplicate matrix coordinate')
        pairs.add((i, j))
        a_ids.update(aa)
        b_ids.update(bb)
        for oid in chain[:-1]:
            intermediate_sizes.add(tensors[outgoing[oid][0]]['size'])
        final_sizes.add(tensors[outgoing[chain[-1]][0]]['size'])
    I, J = len(row_keys), len(col_keys)
    if len(pairs) != I * J or len(ops) != I * J * K:
        raise ValueError('matrix incidence is not Cartesian')
    if len(a_ids) != I*K or len(b_ids) != J*K or a_ids & b_ids:
        raise ValueError('input projection coordinates are not injective')
    a_sizes = {tensors[t]['size'] for t in a_ids}
    b_sizes = {tensors[t]['size'] for t in b_ids}
    if len(a_sizes) != 1 or len(b_sizes) != 1 or len(final_sizes) != 1:
        raise ValueError('nonuniform tensor sizes')
    if len(intermediate_sizes) > 1:
        raise ValueError('nonuniform intermediate sizes')
    sa, sb, sf = next(iter(a_sizes)), next(iter(b_sizes)), next(iter(final_sizes))
    sc = next(iter(intermediate_sizes), sf)
    if min(sa, sb, sc, sf) <= 0 or sf > sc:
        raise ValueError('unsupported accumulator sizes')
    if any(tensors[t]['pos'] != 'L1'
           for t in a_ids | b_ids | {t for o in ops for t in outgoing[o]}):
        raise ValueError('bound requires matmul values in L1')
    unit = gcd(gcd(sa, sb), gcd(sc, sf))
    return dict(I=I, J=J, K=K, products=len(ops), chains=len(chains),
                unit_bytes=unit, wa=sa//unit, wb=sb//unit,
                wc=sc//unit, wf=sf//unit)


@lru_cache(maxsize=None)
def _phase_product_upper(budget, I, J, K, wa, wb, wc):
    """Exact integer maximum of the stated necessary projection inequalities."""
    best = 0
    for a in range(1, min(I*K, (budget-wb-2*wc)//wa) + 1):
        remaining = budget-wa*a
        for b in range(1, min(J*K, (remaining-2*wc)//wb) + 1):
            c = min(I*J, (remaining-wb*b)//(2*wc))
            value = min(isqrt(a*b*c), J*a, I*b, K*c)
            if value > best:
                best = value
    return best


def memory_bounds(graph, capacity_l1, bandwidth, core_counts=(2, 3, 4, 5)):
    """Return per-core certified DDR-cycle lower bounds and integer witnesses."""
    try:
        model = _regular_matmul(graph)
    except ValueError as error:
        return {'applicable': False, 'reason': str(error), 'by_cores': {},
                'cores': {}}
    if capacity_l1 <= 0 or bandwidth <= 0:
        raise ValueError('capacity and bandwidth must be positive')
    I, J, K = (model[x] for x in ('I', 'J', 'K'))
    wa, wb, wc, wf = (model[x] for x in ('wa', 'wb', 'wc', 'wf'))
    unit, N, C = (model[x] for x in ('unit_bytes', 'products', 'chains'))
    M = capacity_l1 // unit
    virtual = (2*wc-wf)*C
    # All real COPY transfers are one of the recognized tensor sizes.
    # A K=1 operation has both virtual charges in the same atomic event.
    max_charge = max(wa, wb, wc, wf,
                     2*wc-wf if K == 1 else wc, wc-wf)
    compulsory_units = wa*I*K + wb*J*K + wf*C
    result = {'applicable': True, 'model': model,
              'capacity_units': M, 'virtual_charge_units': virtual,
              'compulsory_bytes': compulsory_units*unit, 'by_cores': {}}
    for cores in core_counts:
        result['by_cores'][str(cores)] = {
            'io_units_lower_bound': compulsory_units,
            'io_bytes_lower_bound': compulsory_units*unit,
            'cycles_lower_bound': compulsory_units*unit/bandwidth,
            'method': 'compulsory_input_output',
        }
    # q~4M is best in the continuous relaxation. Sweep more broadly because
    # dimensions and integral phase counts materially improve some instances.
    # Enumerate densely through 4M, then sample farther out. Any subset of q
    # retains validity; broader exploration only tightens the resulting bound.
    q_values = sorted(set(range(1, max(128, 4*M)+1)) |
                      set(range(max(128, 4*M)+4, max(256, 12*M)+1, 4)))
    for q in q_values:
        budget = 2*M + q + max_charge - 1
        H = _phase_product_upper(budget, I, J, K, wa, wb, wc)
        if H == 0:
            continue
        phases = (N + H - 1)//H
        for cores in core_counts:
            io_units = max(0, q*max(0, phases-cores)-virtual)
            entry = result['by_cores'][str(cores)]
            if io_units > entry['io_units_lower_bound']:
                entry.update(io_units_lower_bound=io_units,
                             io_bytes_lower_bound=io_units*unit,
                             cycles_lower_bound=io_units*unit/bandwidth,
                             method='finite_memory_projection_phases',
                             q_units=q, phase_boundary_budget_units=budget,
                             max_event_charge_units=max_charge,
                             products_per_phase_upper_bound=H,
                             phases_required_lower_bound=phases)
    for entry in result['by_cores'].values():
        entry['lower_bound'] = ceil(Fraction(entry['io_bytes_lower_bound']) /
                                    Fraction(str(bandwidth)))
    result['cores'] = result['by_cores']
    return result


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--capacity-l1', type=int, default=1572864)
    parser.add_argument('--bandwidth', type=float, default=60)
    parser.add_argument('--output', default='analysis_bounds/memory_bounds.json')
    args = parser.parse_args()
    results = {}
    for path in sorted(Path(args.data_dir).glob('case_*.json')):
        results[path.stem] = memory_bounds(json.loads(path.read_text()),
                                          args.capacity_l1, args.bandwidth)
        print(path.stem, results[path.stem].get('model', {}).get('products'),
              {k: round(v['cycles_lower_bound'], 2)
               for k, v in results[path.stem]['by_cores'].items()}, flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2)+'\n')

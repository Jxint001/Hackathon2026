"""Combine an exact profile certificate with arbitrary short or idle cores.

This file does not assert that the supplied g values have been audited.  The
separate bound_final.py command runs the independent integer oracle first.
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

N, C, P, UNIT, BASELINE = 6144, 192, 5, 32768, 6295267


def combine(a):
    q, den, g = a['q'], a['theta_den'], a['g_num']
    assert a['memory_for_C'] == 39
    assert a['delta_num'] == a['delta_den'] == 1
    assert len(g) == q + 1 and g[0] == 0
    assert sum(a['theta_num']) == 2 * den
    common = q * den
    alpha_num = g[q] - 41 * den
    beta_num = -(q - 1) * g[q] + 2 * sum(g[1:q]) + 41 * den
    # F(n) averages all q offsets, including offsets with no internal cut.
    h = [0]
    prefix = 0
    for n in range(1, q):
        fn = (q - n + 1) * g[n] + 2 * prefix - 41 * den * (n - 1)
        h.append(fn - alpha_num * n)
        prefix += g[n]
    short_dp = [[0]]
    short_witness = [[()]]
    for count in range(1, P):
        values, witnesses = [], []
        old = short_dp[-1]
        for total in range(count * (q - 1) + 1):
            candidates = [(old[total - n] + h[n], n)
                          for n in range(max(0, total - len(old) + 1), min(q - 1, total) + 1)]
            value, last = min(candidates)
            values.append(value)
            witnesses.append(short_witness[-1][total - last] + (last,))
        short_dp.append(values)
        short_witness.append(witnesses)

    def conditional(trial):
        limit = (trial - 1) // 1024
        best = None
        for long_count in range(1, P + 1):
            short_count = P - long_count
            for short_total, short_value in enumerate(short_dp[short_count]):
                remaining = N - short_total
                if long_count * q <= remaining <= long_count * limit:
                    value = alpha_num * N + long_count * beta_num + short_value
                    if best is None or value < best[0]:
                        best = (value, long_count,
                                short_witness[short_count][short_total], remaining)
        if best is None:
            return {'impossible_by_compute': True, 'time': trial, 'core_op_limit': limit}
        # Root-boundary penalty -4C/q.  At least C-P final releases and
        # 1024-2P input last-use releases occur at internal positions.
        raw_num = best[0] + C * common + (1024 - 3 * C - 3 * P) * den
        units = -(-raw_num // common)
        time = (units * UNIT + 59) // 60
        return {'impossible_by_compute': False, 'time': time,
                'core_op_limit': limit, 'raw_io_num': raw_num,
                'raw_io_den': common, 'io_units_ceil': units,
                'io_bytes_lower_bound': units * UNIT,
                'worst_long_core_count': best[1],
                'worst_short_core_lengths': best[2],
                'worst_long_core_op_total': best[3]}

    lo, hi = 0, BASELINE + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if conditional(mid)['time'] >= mid:
            lo = mid
        else:
            hi = mid
    witness = conditional(lo)
    return {'baseline_cycles': BASELINE, 'q': q,
            'alpha': str(Fraction(alpha_num, common)),
            'beta': str(Fraction(beta_num, common)),
            'min_short_h': str(Fraction(min(h), common)),
            'certified_time_lower_bound': lo,
            'certified_speedup_upper_bound': BASELINE / lo,
            'conditional_certificate_at_threshold': witness,
            'next_trial': conditional(hi)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('certificate', type=Path)
    args = ap.parse_args()
    print(json.dumps(combine(json.loads(args.certificate.read_text())), indent=2))

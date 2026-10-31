"""Audit and reproduce the case03 five-core upper bound, without SciPy."""
import hashlib
import json
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from bound_combine import combine

D = Path(__file__).resolve().parent
ROOT = D.parents[3]
a = json.loads((D / 'bound_certificate.json').read_text())
assert a['q'] == 180 and a['theta_den'] == 16
assert a['theta_num'] == [1] * 32
assert a['memory_for_C'] == 39
assert a['delta_num'] == a['delta_den'] == 1
graph_hash = hashlib.sha256((ROOT / 'data/case_03.json').read_bytes()).hexdigest()
assert graph_hash == '2d098e3c81144b9e94e6cd2c8e97325c941112ec50148abd9e6e607bf381fb86'
config_hash = hashlib.sha256((ROOT / 'data/config.txt').read_bytes()).hexdigest()
assert config_hash == '0ebdd000063a14ab81e8f26e07fe5d82baddba865df7e5e64482ba12137df1e7'
inp = '180 39 16 8\n' + ' '.join(map(str, a['theta_num'])) + '\n'
inp += ' '.join(map(str, a['g_num'])) + '\n'
with tempfile.TemporaryDirectory(prefix='case03_bound_') as temp:
    binary = str(Path(temp) / 'oracle')
    subprocess.run(['g++', '-O3', '-std=c++17', str(D / 'bound_oracle_audit.cpp'),
                    '-o', binary], check=True)
    run = subprocess.run([binary], input=inp, text=True, capture_output=True, check=True)
assert 'PASS all 181 g values' in run.stdout

q, N, C, P = 180, 6144, 192, 5
g = [Fraction(v, 16) for v in a['g_num']]
raw = (N * (g[q] - 41) + P * (-(q - 1) * g[q] + 2 * sum(g[1:q]) + 41)) / q
raw += C + Fraction(1024 - 3 * C - 3 * P, q)
assert raw == Fraction(193789, 72)
units = -(-raw.numerator // raw.denominator)
io_time = (units * 32768 + 59) // 60
small_core_time = ((N - (q - 1) + (P - 2)) // (P - 1)) * 1024
assert units == 2692 and io_time == 1470191 and small_core_time == 1527808
out = combine(a)
assert out['certified_time_lower_bound'] == min(io_time, small_core_time)
out.update(status='PASS', graph_sha256=graph_hash, config_sha256=config_hash,
           independent_oracle_audit=run.stdout.strip(),
           io_units_lower_bound_exact=str(raw), io_units_ceil=units,
           io_bytes_lower_bound_when_all_cores_long=units * 32768,
           short_or_idle_core_time_lower_bound=small_core_time,
           scope='All legal five-core plans, arbitrary cross-core chain cuts and subgraph orders')
(D / 'bound_final.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))

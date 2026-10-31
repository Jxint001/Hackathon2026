"""Rebuild the upper-bound table using the fixed evaluator and saved certificates."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(ROOT / 'code'))

from singlecore_evaluate import evaluate_singlecore
from dag_bounds import compute_bounds
from memory_bound import memory_bounds


def certificate_bound(case):
    if case in (3, 5):
        path = DIRECTORY / 'certificates' / f'case{case:02d}' / 'bound_final.json'
    elif case == 8:
        path = DIRECTORY / 'certificates/case08/structure_upper_certificate.json'
    else:
        return 0
    certificate = json.loads(path.read_text(encoding='utf-8'))
    for key, source in (
        ('graph_sha256', ROOT / f'data/case_{case:02d}.json'),
        ('config_sha256', ROOT / 'data/config.txt'),
    ):
        if key in certificate:
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != certificate[key]:
                raise ValueError(f'Certificate input has changed: {source}')
    return certificate.get('certified_time_lower_bound', certificate.get('time_lower_bound', 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, nargs='+', choices=range(10), default=list(range(10)))
    parser.add_argument('-o', '--output', type=Path, default=DIRECTORY / 'table.json')
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        graph = json.loads((ROOT / f'data/case_{case:02d}.json').read_text(encoding='utf-8'))
        specialized = certificate_bound(case)
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = evaluate_singlecore(graph, 60, {'L1': 1572864, 'UB': 131072})['makespan']
        dag = compute_bounds(graph, 60, core_counts=(2, 3, 4, 5))
        memory = memory_bounds(graph, 1572864, 60)
        row = {'case': f'case{case:02d}', 'baseline': baseline, 'bounds': {}}
        for cores in range(2, 6):
            lower = dag['cores'][str(cores)]['lower_bound']
            if memory.get('applicable'):
                lower = max(lower, memory['by_cores'][str(cores)]['lower_bound'])
            # A five-core lower bound also covers fewer cores, since idle cores are legal.
            lower = max(lower, specialized)
            row['bounds'][str(cores)] = {
                'time_lower_bound': lower,
                'speedup_upper_bound': baseline / lower,
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

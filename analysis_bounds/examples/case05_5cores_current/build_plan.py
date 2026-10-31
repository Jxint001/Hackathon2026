#!/usr/bin/env python3
"""Rebuild the verified case05 five-core construction from its panel recipe."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from matmul_plans import infer_structure,validate_structure_plan

def construct(graph,recipe):
    s=infer_structure(graph)
    orders=[]
    for panels in recipe:
        orders.append([s.chains[row,column][k]
                       for panel in panels
                       for column in panel['columns']
                       for k in range(s.depth)
                       for row in panel['rows']])
    plan={'node_to_subgraph':{str(op):op for op in sorted(s.compute_ids)},
          'core_schedules':orders}
    validate_structure_plan(s,plan)
    return plan

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--graph',type=Path,default=ROOT/'data/case_05.json')
    ap.add_argument('--recipe',type=Path,default=Path(__file__).with_name('recipe.json'))
    ap.add_argument('-o','--output',type=Path,default=Path(__file__).with_name('plan.json'))
    args=ap.parse_args()
    plan=construct(json.loads(args.graph.read_text()),json.loads(args.recipe.read_text()))
    args.output.write_text(json.dumps(plan)+'\n')
    print(args.output)

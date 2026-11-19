# -*- coding: utf-8 -*-
"""Per-mutant detail used in the discussion: S2 mutants missed per configuration."""
import json, glob, numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
from collections import Counter
from s1 import ref_out
D = os.path.join(HERE, 'data', '')
lab = {str(r['id']): r for r in json.load(open(D + 's2_labels.json'))}
ids = json.load(open(D + 'suites_all_s2.out.ids.json'))
allv = np.fromfile(D + 'suites_all.bin', np.int8).reshape(-1, 21)
O = np.fromfile(D + 'suites_all_s2.out', np.int8).reshape(-1, len(allv), 5)
R = ref_out(allv); idx = {v.tobytes(): i for i, v in enumerate(allv)}
K = (O[1:] != R[None]).any(2)
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, 'runs', '*.json')))]
B0 = np.load(D + 'B0.npy').astype(np.int8)
out = {'kill_121': Counter(), 'missed_all_seeds': {}}
miss = {}
for r in runs:
    vi = [idx[np.array(v, np.int8).tobytes()] for v in r['suite']]
    for k, m in enumerate(ids):
        if lab[str(m['id'])]['equivalent']:
            continue
        killed = bool(K[k][vi].any())
        if str(m['id']) == '121':
            out['kill_121'][r['config']] += int(killed)
        miss.setdefault(r['config'], Counter())[str(m['id'])] += int(not killed)
for c, cnt in miss.items():
    allm = [i for i, n in cnt.items() if n == 30]
    out['missed_all_seeds'][c] = dict(ids=allm, families=dict(Counter(lab[i]['family'] for i in allm)))
b0i = [idx[v.tobytes()] for v in B0]
out['b0_missed'] = [str(m['id']) for k, m in enumerate(ids) if not lab[str(m['id'])]['equivalent'] and not K[k][b0i].any()]
json.dump(out, open(D + 'extra.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# -*- coding: utf-8 -*-
"""Per-mutant detail for the LLM configurations (mutant 121 and S2 mutants missed on every seed).
Needs the S2 output of the offline re-measurement (data/llm_all.bin, data/llm_all_s2.out)."""
import json, glob, os, sys
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from s1 import ref_out
D = os.path.join(ROOT, 'data', ''); OUT = os.path.join(ROOT, 'results_llm', '')
lab = {str(r['id']): r for r in json.load(open(D + 's2_labels.json'))}
ids = json.load(open(D + 'llm_all_s2.out.ids.json'))
allv = np.fromfile(D + 'llm_all.bin', np.int8).reshape(-1, 21)
O = np.fromfile(D + 'llm_all_s2.out', np.int8).reshape(-1, len(allv), 5)
R = ref_out(allv); idx = {v.tobytes(): i for i, v in enumerate(allv)}
K = (O[1:] != R[None]).any(2)
runs = [json.load(open(f)) for f in sorted(glob.glob(OUT + 'runs/*.json'))]
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
# valid per-seed LLM tests (seeds 1 to 30, temperature 0.7): do they reach the condition of mutant 121?
sys.path.insert(0, HERE)
from llm_experiment import validate_group
rows = [json.loads(l) for l in open(OUT + 'llm_log.jsonl', encoding='utf-8')]
seeds = sorted({r['seed'] for r in rows if r['temperature'] == 0.7})
vecs, t_flag = [], []
for s in seeds:
    for r in validate_group([r for r in rows if r['temperature'] == 0.7 and r['seed'] == s]):
        if r['valid']:
            vecs.append(r['record']['items'])
            if r['target_index'] == 4:
                t_flag.append(bool(r['record']['target_met']))
X = np.array(vecs, np.int8); Ov = ref_out(X); crit = (X[:, [9, 16, 20]] == 3).any(1)
out['valid_llm_tests_seeds_1_30'] = dict(n=len(X), depression_category_counts=np.bincount(Ov[:, 0], minlength=5).tolist(),
                                         severe_or_above_without_critical=int(((Ov[:, 0] >= 3) & ~crit).sum()),
                                         extremely_severe_without_critical=int(((Ov[:, 0] == 4) & ~crit).sum()),
                                         flag_target_valid=len(t_flag), flag_target_met=sum(t_flag))
json.dump(out, open(OUT + 'extra_llm.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

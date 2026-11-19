# -*- coding: utf-8 -*-
"""Step 2 (S2): detect killable StrykerJS mutants on a large structured input set; the
survivors are reviewed manually for equivalence on the valid input domain {0..3}^21."""
import numpy as np, subprocess, json, itertools
import os
HERE = os.path.dirname(os.path.abspath(__file__))
from s1 import *
D = os.path.join(HERE, 'data', '')
rng = np.random.default_rng(2026)
vs = []
# (a) every lattice triple with rotating construction k = 0..6 (item placement varies)
for rd, ra, rs in itertools.product(range(22), repeat=3):
    for k in range(7):
        perm = [((j + k) % 7) for j in range(7)]
        vs.append(construct([rd, ra, rs], [perm, perm, perm]))
# (b) critical-item conditions: each critical item at 0..3 with D Normal, other items free
for _ in range(100000):
    x = rng.integers(0, 4, 21)
    vs.append(list(x))
# (c) uniform over raw subscale sums with random placement
for _ in range(200000):
    raw = rng.integers(0, 22, 3); perms = [list(rng.permutation(7)) for _ in range(3)]
    vs.append(construct(list(raw), perms))
X = np.array(vs, np.int8)
X = np.unique(X, axis=0)
X.tofile(D + 'equiv_s2.bin')
print('structured equivalence set', X.shape)
subprocess.run(['node', os.path.join(HERE, 's2', 'eval_mutants.js'), os.path.join(HERE, 's2', 'reports', 'mutation.json'), D + 'equiv_s2.bin', D + 'equiv_s2_kill.json', 'kill'], check=True)
k = json.load(open(D + 'equiv_s2_kill.json'))
ids = json.load(open(D + 'b0_s2.out.ids.json'))
surv = [m for m, s in zip(ids, k) if s['kills'] == 0]
print('killed', sum(1 for s in k if s['kills'] > 0), 'survivors', len(surv))
for m in surv:
    print(m['id'], m['mutator'], 'L%d:%d' % (m['line'], m['col']), repr(m['original'][:70]), '->', repr(m['replacement'][:70]))

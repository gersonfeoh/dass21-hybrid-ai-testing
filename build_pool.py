# -*- coding: utf-8 -*-
"""Step 2 and 3 of the protocol: build the S1 mutant pool, verify the vectorised
evaluator against the Article 1 reference engine, mark equivalent mutants with Z3,
sample second-order mutants, and split the pool by fault site (seed 2026)."""
import sys, json, random, time, csv, itertools
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'article1'))
import mesin
from s1 import *

OUT = os.path.join(HERE, 'data', '')
rng = np.random.default_rng(12345)

# 1. evaluator check against the Article 1 reference engine and the 35-vector oracle
X = rng.integers(0, 4, size=(200000, 21))
ref = ref_out(X)
acuan = mesin.Mesin()
sample = X[:20000]
exp = np.array([acuan(list(map(int, x))) for x in sample], dtype=np.int8)
assert (ref[:20000] == exp).all(), 'vectorised reference differs from mesin.Mesin'
EN = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely Severe': 4}
rows = list(csv.DictReader(open(os.path.join(HERE, 'article1', 'vektor_uji_bva_kisi_putar.csv'))))
B0 = np.array([[int(r['b%02d' % i]) for i in range(1, 22)] for r in rows], np.int8)
B0_exp = np.array([[EN[r['harapan_D']], EN[r['harapan_A']], EN[r['harapan_S']],
                    r['harapan_gerbang'] == 'terbuka', r['harapan_penanda'] == 'ya'] for r in rows], np.int8)
assert (ref_out(B0) == B0_exp).all()
np.save(OUT + 'B0.npy', B0)
print('reference evaluator verified on 20,000 random vectors and the 35-vector oracle')

# 2. first-order pool
fo = first_order()
fam_count = {}
for m in fo:
    fam_count[m['family']] = fam_count.get(m['family'], 0) + 1
print('first-order mutants', len(fo), fam_count)

# cross-check every first-order mutant that also exists in Article 1 against mesin.mutan()
art1 = mesin.mutan()
def mesin_params(m):
    p = ref_params()
    p['assign'] = [SUB.index(next(s for s in SUB if i in m.butir[s])) for i in range(1, 22)]
    p['factor'] = [m.faktor] * 3
    p['thr'] = [list(m.batas[s]) for s in SUB]
    p['g_comb'] = {'semua_normal': 'all', 'salah_satu_normal': 'any', 'maks_mild': 'all'}[m.gerbang]
    p['g_level'] = 1 if m.gerbang == 'maks_mild' else 0
    p['g_subs'] = [int(s in m.gerbang_sub) for s in SUB]
    p['f_min'] = m.penanda_min; p['f_subs'] = [int(s in m.penanda_sub) for s in SUB]
    p['f_comb'] = {'atau': 'or', 'dan': 'and', 'kategori_saja': 'cat', 'butir_saja': 'item'}[m.penanda_gabung]
    p['f_items'] = list(m.butir_kritis); p['f_item_min'] = m.butir_min
    return p
key = lambda p: json.dumps(p, sort_keys=True)
fo_by_key = {key(m['params']): m for m in fo}
for kelas, m in art1:
    p = mesin_params(m)
    assert key(p) in fo_by_key, m.nama
    fo_by_key[key(p)]['article1'] = True
    out = Pool([p]).run(sample[:3000])[0]
    exp = np.array([m(list(map(int, x))) for x in sample[:3000]], dtype=np.int8)
    assert (out == exp).all(), m.nama
print('all 105 Article 1 mutants are contained in the pool and evaluate identically')

# 3. exact equivalence with Z3
t0 = time.time()
for m in fo:
    eq, w = z3_equivalent(m['params'])
    m['equivalent'] = eq; m['witness'] = w
    if not eq:  # witness must really distinguish
        assert (Pool([m['params']]).run([w])[0] != ref_out([w])).any()
print('Z3 equivalence on first-order pool: %d equivalent, %.1f s' % (sum(m['equivalent'] for m in fo), time.time() - t0))
a1 = [m for m in fo if m.get('article1')]
print('Article 1 subset: %d mutants, %d equivalent (Article 1 reported 12)' % (len(a1), sum(m['equivalent'] for m in a1)))

# 4. second-order sample from pairs of the non-equivalent Article 1 mutants
a1ne = [m for m in a1 if not m['equivalent']]
pairs = list(itertools.combinations(range(len(a1ne)), 2))
print('pairs of non-equivalent Article 1 mutants:', len(pairs))
strata = {}
for i, j in pairs:
    p = compose(a1ne[i]['params'], a1ne[j]['params'])
    if p is None:
        continue
    fam = tuple(sorted((a1ne[i]['family'], a1ne[j]['family'])))
    strata.setdefault(fam, []).append((i, j, p))
print('compatible pairs per stratum:', {'+'.join(k): len(v) for k, v in strata.items()})
r = random.Random(2026)
TARGET = 500
alloc = {k: 0 for k in strata}
left = TARGET
while left > 0:  # equal allocation, capped by availability
    open_ = [k for k in strata if alloc[k] < len(strata[k])]
    share = max(1, left // len(open_))
    for k in open_:
        add = min(share, len(strata[k]) - alloc[k], left)
        alloc[k] += add; left -= add
        if left == 0:
            break
so = []
for k in sorted(strata):
    for i, j, p in r.sample(strata[k], alloc[k]):
        m1, m2 = a1ne[i], a1ne[j]
        fam = m1['family'] if m1['family'] == m2['family'] else 'mixed'
        so.append(dict(family=fam, site='%s|%s' % tuple(sorted((m1['site'], m2['site']))), name='%s + %s' % (m1['name'], m2['name']),
                       order=2, params=p, parents=[m1['id'], m2['id']]))
t0 = time.time()
for n, m in enumerate(so):
    m['id'] = 'S1-2-%03d' % (n + 1)
    eq, w = z3_equivalent(m['params'])
    m['equivalent'] = eq; m['witness'] = w
print('second-order sample %d (allocation %s); %d equivalent; %.1f s' % (len(so), {'+'.join(k): v for k, v in alloc.items()}, sum(m['equivalent'] for m in so), time.time() - t0))

# 5. split by fault site, 70% training / 30% evaluation, stratified by family, seed 2026
r = random.Random(2026)
site_split = {}
for fam in ('threshold', 'scaling', 'mapping', 'gate', 'flag'):
    sites = sorted({m['site'] for m in fo if m['family'] == fam and not m['equivalent']})
    r.shuffle(sites)
    k = max(1, round(0.3 * len(sites)))
    for s in sites[:k]:
        site_split[s] = 'eval'
    for s in sites[k:]:
        site_split[s] = 'train'
for m in fo:
    m['pool'] = 'equivalent' if m['equivalent'] else site_split[m['site']]
for m in so:
    if m['equivalent']:
        m['pool'] = 'equivalent'
    else:
        a, b = (site_split.get(s) for s in m['site'].split('|'))
        m['pool'] = a if a == b else 'cross'
json.dump(dict(first_order=fo, second_order=so, site_split=site_split), open(OUT + 'mutants_S1.json', 'w'), indent=0)
with open(OUT + 'mutants_S1.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'order', 'family', 'site', 'name', 'article1', 'equivalent', 'pool'])
    for m in fo + so:
        w.writerow([m['id'], m['order'], m['family'], m['site'], m['name'], int(bool(m.get('article1'))), int(m['equivalent']), m['pool']])
summ = {}
for m in fo + so:
    summ.setdefault((m['order'], m['family'], m['pool']), 0)
    summ[(m['order'], m['family'], m['pool'])] += 1
for k in sorted(summ):
    print(k, summ[k])

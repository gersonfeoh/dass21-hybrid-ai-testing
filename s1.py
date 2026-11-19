# -*- coding: utf-8 -*-
"""Subject S1: parametric DASS-21 reference engine (Article 1) with an extended
specification-level mutant pool, a vectorised evaluator, and exact equivalence
checking with Z3.

Outputs per response vector: (c_D, c_A, c_S, G, F), categories 0..4.
"""
import itertools, random, json
import numpy as np

SUB = ('D', 'A', 'S')
BUTIR = {'D': (3, 5, 10, 13, 16, 17, 21), 'A': (2, 4, 7, 9, 15, 19, 20), 'S': (1, 6, 8, 11, 12, 14, 18)}
BATAS = {'D': (10, 14, 21, 28), 'A': (8, 10, 15, 20), 'S': (15, 19, 26, 34)}
KRITIS = (10, 17, 21)
LAMBDA = list(range(0, 43, 2))


def ref_params():
    return dict(assign=[SUB.index(next(s for s in SUB if i in BUTIR[s])) for i in range(1, 22)],
                factor=[2, 2, 2], thr=[list(BATAS[s]) for s in SUB],
                g_comb='all', g_level=0, g_subs=[1, 1, 1],
                f_min=3, f_subs=[1, 0, 0], f_comb='or', f_items=list(KRITIS), f_item_min=3)


# ------------------------------------------------------------------ mutant pool
def first_order():
    """First-order specification-level mutants: (family, site, name, params)."""
    out = []
    R = ref_params

    def add(fam, site, name, **ch):
        p = R(); p.update({k: v for k, v in ch.items()})
        out.append(dict(family=fam, site=site, name=name, order=1, params=p))

    # threshold: every cut-point shifted by delta in {+-1..+-4}, keeping order
    for si, s in enumerate(SUB):
        for j in range(4):
            for d in (-4, -3, -2, -1, 1, 2, 3, 4):
                t = [list(BATAS[u]) for u in SUB]; t[si][j] += d
                if (j > 0 and t[si][j] <= t[si][j - 1]) or (j < 3 and t[si][j] >= t[si][j + 1]):
                    continue
                add('threshold', 'thr:%s%d' % (s, j + 1), '%s cut %d %+d' % (s, j + 1, d), thr=t)
    # scaling
    for f in (1, 3, 4):
        add('scaling', 'scale:all', 'factor %d on all subscales' % f, factor=[f, f, f])
    for si, s in enumerate(SUB):
        for f in (1, 4):
            fac = [2, 2, 2]; fac[si] = f
            add('scaling', 'scale:%s' % s, 'factor %d on %s only' % (f, s), factor=fac)
    # item mapping: one item counted on another subscale
    for i in range(1, 22):
        own = R()['assign'][i - 1]
        for t in range(3):
            if t == own:
                continue
            a = R()['assign']; a[i - 1] = t
            add('mapping', 'item:%d' % i, 'item %d %s to %s' % (i, SUB[own], SUB[t]), assign=a)
    # gate
    add('gate', 'gate:comb', 'gate OR over subscales', g_comb='any')
    add('gate', 'gate:comb', 'gate always open', g_comb='true')
    add('gate', 'gate:polarity', 'gate negated', g_comb='nall')
    add('gate', 'gate:level', 'gate up to Mild', g_level=1)
    add('gate', 'gate:level', 'gate up to Moderate', g_level=2)
    for si, s in enumerate(SUB):
        m = [1, 1, 1]; m[si] = 0
        add('gate', 'gate:omit%s' % s, 'gate ignores %s' % s, g_subs=m)
    for si, s in enumerate(SUB):
        m = [0, 0, 0]; m[si] = 1
        add('gate', 'gate:only%s' % s, 'gate uses %s only' % s, g_subs=m)
    # flag
    add('flag', 'flag:catlevel', 'flag category from Moderate', f_min=2)
    add('flag', 'flag:catlevel', 'flag category Extremely Severe only', f_min=4)
    add('flag', 'flag:catsub', 'flag category on any subscale', f_subs=[1, 1, 1])
    add('flag', 'flag:catsub', 'flag category on D or A', f_subs=[1, 1, 0])
    add('flag', 'flag:catsub', 'flag category on D or S', f_subs=[1, 0, 1])
    add('flag', 'flag:catsub', 'flag category on A instead of D', f_subs=[0, 1, 0])
    add('flag', 'flag:catsub', 'flag category on S instead of D', f_subs=[0, 0, 1])
    add('flag', 'flag:comb', 'flag conjunction', f_comb='and')
    add('flag', 'flag:comb', 'flag without item rule', f_comb='cat')
    add('flag', 'flag:comb', 'flag without category rule', f_comb='item')
    add('flag', 'flag:itemmin', 'flag item value from 2', f_item_min=2)
    for i in KRITIS:
        add('flag', 'flag:item%d' % i, 'flag ignores item %d' % i, f_items=[j for j in KRITIS if j != i])
    for i in KRITIS:
        add('flag', 'flag:item%d' % i, 'flag reads item %d instead of %d' % (i - 1, i),
            f_items=[(j - 1 if j == i else j) for j in KRITIS])
    for i, k in enumerate(out):
        k['id'] = 'S1-%03d' % (i + 1)
    return out


def diff_keys(p):
    r = ref_params()
    return [k for k in r if p[k] != r[k]]


def compose(m1, m2):
    """Second-order mutant from two parameter dicts: apply both changes; None if they touch the same parameter key
    in incompatible ways (assign and thr are merged element-wise)."""
    r = ref_params(); p = ref_params()
    for m in (m1, m2):
        for k in diff_keys(m):
            v = m[k]
            if k == 'assign':
                for i, (a, b) in enumerate(zip(v, r[k])):
                    if a != b:
                        if p[k][i] != r[k][i]:
                            return None
                        p[k][i] = a
            elif k == 'thr':
                for si in range(3):
                    for j in range(4):
                        if v[si][j] != r[k][si][j]:
                            if p[k][si][j] != r[k][si][j]:
                                return None
                            p[k][si][j] = v[si][j]
                for si in range(3):
                    if any(p['thr'][si][j] >= p['thr'][si][j + 1] for j in range(3)):
                        return None
            else:
                if p[k] != r[k]:
                    return None
                p[k] = v
    return p


# ------------------------------------------------------------------ vectorised evaluation
COMB_G = {'all': 0, 'any': 1, 'true': 2, 'nall': 3}
COMB_F = {'or': 0, 'and': 1, 'cat': 2, 'item': 3}


class Pool:
    """Stacked parameters of a list of engines for vectorised evaluation."""

    def __init__(self, params_list):
        M = len(params_list)
        self.M = M
        self.A = np.zeros((M, 21, 3), np.int16)
        self.fac = np.zeros((M, 3), np.int16)
        self.thr = np.zeros((M, 3, 4), np.int16)
        self.gc = np.zeros(M, np.int8); self.gl = np.zeros(M, np.int16); self.gs = np.zeros((M, 3), bool)
        self.fm = np.zeros(M, np.int16); self.fs = np.zeros((M, 3), bool); self.fc = np.zeros(M, np.int8)
        self.fi = np.zeros((M, 21), bool); self.fim = np.zeros(M, np.int16)
        for m, p in enumerate(params_list):
            for i, a in enumerate(p['assign']):
                self.A[m, i, a] = 1
            self.fac[m] = p['factor']; self.thr[m] = p['thr']
            self.gc[m] = COMB_G[p['g_comb']]; self.gl[m] = p['g_level']; self.gs[m] = np.array(p['g_subs'], bool)
            self.fm[m] = p['f_min']; self.fs[m] = np.array(p['f_subs'], bool); self.fc[m] = COMB_F[p['f_comb']]
            for i in p['f_items']:
                self.fi[m, i - 1] = True
            self.fim[m] = p['f_item_min']

    def run(self, X):
        """X: (n,21) int -> outputs (M,n,5) int8."""
        X = np.asarray(X, np.int16)
        raw = np.einsum('nk,mks->mns', X, self.A)
        score = raw * self.fac[:, None, :]
        cat = (score[..., None] >= self.thr[:, None, :, :]).sum(-1)
        ok = cat <= self.gl[:, None, None]
        g_all = (ok | ~self.gs[:, None, :]).all(-1)
        g_any = (ok & self.gs[:, None, :]).any(-1)
        gc = self.gc[:, None]
        G = np.where(gc == 0, g_all, np.where(gc == 1, g_any, np.where(gc == 2, True, ~g_all)))
        pk = ((cat >= self.fm[:, None, None]) & self.fs[:, None, :]).any(-1)
        kr = ((X[None, :, :] >= self.fim[:, None, None]) & self.fi[:, None, :]).any(-1)
        fc = self.fc[:, None]
        F = np.where(fc == 0, pk | kr, np.where(fc == 1, pk & kr, np.where(fc == 2, pk, kr)))
        out = np.empty(cat.shape[:2] + (5,), np.int8)
        out[..., :3] = cat; out[..., 3] = G; out[..., 4] = F
        return out


REF = Pool([ref_params()])


def ref_out(X):
    return REF.run(X)[0]


# ------------------------------------------------------------------ Z3 model
def z3_engine(p, x):
    """Symbolic outputs of an engine with parameters p for symbolic item vector x."""
    import z3
    sc = []
    for si in range(3):
        sc.append(p['factor'][si] * z3.Sum([x[i] for i in range(21) if p['assign'][i] == si]))
    cat = [z3.Sum([z3.If(sc[si] >= t, 1, 0) for t in p['thr'][si]]) for si in range(3)]
    ok = [cat[si] <= p['g_level'] for si in range(3)]
    sel = [si for si in range(3) if p['g_subs'][si]]
    g_all = z3.And([ok[si] for si in sel]) if sel else z3.BoolVal(True)
    g_any = z3.Or([ok[si] for si in sel]) if sel else z3.BoolVal(False)
    G = {'all': g_all, 'any': g_any, 'true': z3.BoolVal(True), 'nall': z3.Not(g_all)}[p['g_comb']]
    fsel = [si for si in range(3) if p['f_subs'][si]]
    pk = z3.Or([cat[si] >= p['f_min'] for si in fsel]) if fsel else z3.BoolVal(False)
    kr = z3.Or([x[i - 1] >= p['f_item_min'] for i in p['f_items']]) if p['f_items'] else z3.BoolVal(False)
    F = {'or': z3.Or(pk, kr), 'and': z3.And(pk, kr), 'cat': pk, 'item': kr}[p['f_comb']]
    return cat, G, F, sc


def z3_vars():
    import z3
    x = [z3.Int('x%d' % (i + 1)) for i in range(21)]
    dom = [z3.And(v >= 0, v <= 3) for v in x]
    return x, dom


def z3_equivalent(p, ref=None):
    """Exact equivalence on the full domain {0..3}^21. Returns (equivalent, witness)."""
    import z3
    ref = ref or ref_params()
    x, dom = z3_vars()
    c1, g1, f1, _ = z3_engine(ref, x)
    c2, g2, f2, _ = z3_engine(p, x)
    s = z3.Solver()
    s.add(dom)
    s.add(z3.Or(c1[0] != c2[0], c1[1] != c2[1], c1[2] != c2[2], g1 != g2, f1 != f2))
    r = s.check()
    if r == z3.unsat:
        return True, None
    assert r == z3.sat
    mdl = s.model()
    return False, [mdl.eval(v, model_completion=True).as_long() for v in x]


# ------------------------------------------------------------------ construction helpers
def construct(raw, perms):
    """Lattice-aware decoder: raw sums per subscale (0..21) and an item-filling order per
    subscale (permutation of the 7 positions) -> item vector."""
    x = [0] * 21
    for si, s in enumerate(SUB):
        r = raw[si]
        for pos in perms[si]:
            i = BUTIR[s][pos]
            v = min(3, r); x[i - 1] = v; r -= v
            if r == 0:
                break
    return x

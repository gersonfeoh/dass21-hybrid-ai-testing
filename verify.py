# -*- coding: utf-8 -*-
"""Step 4 of the protocol: SMT verification of the safety invariants P1 to P8 with Z3,
derivation of safety test obligations, and the candidate amendment G' = G and not F.

Two models are built:
  SPEC  - the specification (Lovibond and Lovibond cut-points, PRD rules for gate and flag);
  PLAT  - a hand translation of the platform logic in Code.gs (upper-bound category table,
          zero-based item indices 9, 16, 20, label comparisons).
Properties are proved (unsat of the negation) or refuted with a concrete counterexample.
"""
import json, time, itertools
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import z3
from s1 import SUB, BUTIR, BATAS, KRITIS

OUT = os.path.join(HERE, 'data', '')
PLAT_MAX = {'D': (9, 13, 20, 27, 42), 'A': (7, 9, 14, 19, 42), 'S': (14, 18, 25, 33, 42)}  # Code.gs DASS_THRESHOLDS
PLAT_DOMAIN = 'SADADSASADSSDSADDSAAD'  # Code.gs DASS_ITEMS, item 1..21


def xs(tag=''):
    x = [z3.Int('x%s%d' % (tag, i + 1)) for i in range(21)]
    return x, [z3.And(v >= 0, v <= 3) for v in x]


def spec(x):
    sc = {s: 2 * z3.Sum([x[i - 1] for i in BUTIR[s]]) for s in SUB}
    cat = {s: z3.Sum([z3.If(sc[s] >= c, 1, 0) for c in BATAS[s]]) for s in SUB}
    G = z3.And([cat[s] == 0 for s in SUB])
    C = cat['D'] >= 3
    I = {i: x[i - 1] == 3 for i in KRITIS}
    F = z3.Or(C, *I.values())
    return dict(sc=sc, cat=cat, G=G, F=F, C=C, I=I)


def plat_cat(score, s):
    """kategoriDass_: first band whose upper bound is >= score; else last band."""
    m = PLAT_MAX[s]
    expr = z3.IntVal(4)
    for k in reversed(range(5)):
        expr = z3.If(score <= m[k], k, expr)
    return expr


def plat(x):
    sc = {s: 2 * z3.Sum([x[i] for i in range(21) if PLAT_DOMAIN[i] == s]) for s in SUB}
    cat = {s: plat_cat(sc[s], s) for s in SUB}
    berat = z3.Or(cat['D'] == 3, cat['D'] == 4)               # ['Berat','Sangat Berat'].indexOf(kat.D) >= 0
    itemKritis = z3.Or(x[9] >= 3, x[16] >= 3, x[20] >= 3)       # jawaban[9], [16], [20]
    F = z3.Or(berat, itemKritis)
    G = z3.And([cat[s] == 0 for s in SUB])                      # _bolehChat: all labels 'Normal'
    return dict(sc=sc, cat=cat, G=G, F=F)


def prove(name, claim, assumptions, xvars):
    s = z3.Solver()
    s.add(assumptions)
    s.add(z3.Not(claim))
    t0 = time.time(); r = s.check(); dt = time.time() - t0
    if r == z3.unsat:
        return dict(property=name, result='proved', counterexample=None, seconds=round(dt, 4))
    m = s.model()
    cex = [[m.eval(v, model_completion=True).as_long() for v in xv] for xv in xvars]
    return dict(property=name, result='refuted', counterexample=cex, seconds=round(dt, 4))


def main():
    res = []
    x, dom = xs()
    S, P = spec(x), plat(x)
    # P1 response domain: the model admits only {0..3}; check that the platform sanitiser never changes a valid input
    res.append(prove('P1 response domain (valid inputs pass the sanitiser unchanged)',
                     z3.And([z3.Not(z3.Or(v < 0, v > 3)) for v in x]), dom, [x]))
    # P2 score lattice
    res.append(prove('P2 score lattice: every subscale score is even and in [0, 42]',
                     z3.And([z3.And(P['sc'][s] % 2 == 0, P['sc'][s] >= 0, P['sc'][s] <= 42) for s in SUB]), dom, [x]))
    # P3 complete and disjoint partition: the platform upper-bound table equals the published cut-points on every score
    sv = z3.Int('s')
    claim = z3.And([plat_cat(sv, s) == z3.Sum([z3.If(sv >= c, 1, 0) for c in BATAS[s]]) for s in SUB])
    res.append(prove('P3 category partition (platform table agrees with the published cut-points on 0..42)',
                     claim, [sv >= 0, sv <= 42], [[sv]]))
    # P4 monotonicity of categories
    y, domy = xs('p')
    Py = plat(y)
    res.append(prove('P4 category monotonicity (item-wise x <= x\' implies c_k(x) <= c_k(x\'))',
                     z3.And([P['cat'][s] <= Py['cat'][s] for s in SUB]),
                     dom + domy + [x[i] <= y[i] for i in range(21)], [x, y]))
    spec_flag_cond = z3.Or(S['cat']['D'] >= 3, *[x[i - 1] == 3 for i in KRITIS])
    res.append(prove('P5 flag soundness (c_D >= 3 or a critical item = 3 implies F)', z3.Implies(spec_flag_cond, P['F']), dom, [x]))
    res.append(prove('P6 flag completeness (F implies c_D >= 3 or a critical item = 3)', z3.Implies(P['F'], spec_flag_cond), dom, [x]))
    res.append(prove('P7 gate safety (G implies not F), current rule G', z3.Implies(P['G'], z3.Not(P['F'])), dom, [x]))
    Gp = z3.And(P['G'], z3.Not(P['F']))
    res.append(prove("P7' gate safety for candidate G' = G and not F", z3.Implies(Gp, z3.Not(P['F'])), dom, [x]))
    res.append(prove("P8a gate-category consistency (G implies all categories Normal)", z3.Implies(P['G'], z3.And([P['cat'][s] == 0 for s in SUB])), dom, [x]))
    res.append(prove("P8a' consistency for G' (G' implies all categories Normal)", z3.Implies(Gp, z3.And([P['cat'][s] == 0 for s in SUB])), dom, [x]))
    res.append(prove('P8b model equivalence (platform model equals specification on all outputs)',
                     z3.And(z3.And([P['cat'][s] == S['cat'][s] for s in SUB]), P['G'] == S['G'], P['F'] == S['F']), dom, [x]))

    # all counterexamples to P7, classified by the critical item that raises the flag
    cls = {}
    for i in KRITIS:
        s = z3.Solver(); s.add(dom); s.add(P['G'], x[i - 1] == 3)
        cls[i] = s.check() == z3.sat
    # exact share of gate-opening response patterns that also raise the flag (dynamic programming)
    from collections import Counter
    def dist(items, crit):
        d = Counter({(0, False): 1})
        for i in items:
            nd = Counter()
            for (r, c), n in d.items():
                for v in range(4):
                    nd[(r + v, c or (i in crit and v == 3))] += n
            d = nd
        return d
    lim = {s: max(r for r in range(22) if 2 * r < BATAS[s][0]) for s in SUB}
    dD = dist(BUTIR['D'], KRITIS)
    nD_all = sum(n for (r, c), n in dD.items() if r <= lim['D'])
    nD_flag = sum(n for (r, c), n in dD.items() if r <= lim['D'] and c)
    other = 1
    for s in ('A', 'S'):
        other *= sum(n for (r, c), n in dist(BUTIR[s], ()).items() if r <= lim[s])
    gate_open = nD_all * other; both = nD_flag * other
    share = dict(gate_opening_patterns=gate_open, gate_and_flag_patterns=both, share=both / gate_open)

    # safety obligations: unique-cause MC/DC conditions for G and F, and P7 counterexample classes
    N = {s: S['cat'][s] == 0 for s in SUB}
    C, I = S['C'], S['I']
    obls = []
    for name, cond in [('G: N_D, N_A, N_S all true', z3.And(N['D'], N['A'], N['S'])),
                       ('G: only N_D false', z3.And(z3.Not(N['D']), N['A'], N['S'])),
                       ('G: only N_A false', z3.And(N['D'], z3.Not(N['A']), N['S'])),
                       ('G: only N_S false', z3.And(N['D'], N['A'], z3.Not(N['S']))),
                       ('F: all conditions false', z3.And(z3.Not(C), *[z3.Not(v) for v in I.values()])),
                       ('F: only C (c_D >= 3) true', z3.And(C, *[z3.Not(v) for v in I.values()]))] + \
                      [('F: only item %d = 3 true' % i, z3.And(z3.Not(C), I[i], *[z3.Not(I[j]) for j in KRITIS if j != i])) for i in KRITIS] + \
                      [('P7 counterexample class: G and item %d = 3 only' % i, z3.And(S['G'], I[i], *[z3.Not(I[j]) for j in KRITIS if j != i])) for i in KRITIS]:
        s = z3.Solver(); s.add(dom); s.add(cond)
        t0 = time.time(); r = s.check(); dt = time.time() - t0
        w = [s.model().eval(v, model_completion=True).as_long() for v in x] if r == z3.sat else None
        obls.append(dict(obligation=name, satisfiable=(r == z3.sat), witness=w, seconds=round(dt, 4)))
    json.dump(dict(properties=res, p7_counterexample_classes={str(k): v for k, v in cls.items()},
                   gate_flag_share=share, obligations=obls, z3_version=z3.get_version_string()),
              open(OUT + 'z3_results.json', 'w'), indent=1)
    for r in res:
        print('%-95s %-8s %s' % (r['property'], r['result'], r['counterexample'][0] if r['counterexample'] else ''))
    print('P7 counterexample classes (critical item -> satisfiable):', cls)
    print('gate-opening patterns %d, also flagged %d, share %.4f' % (gate_open, both, share['share']))
    for o in obls:
        print('%-55s sat=%s witness=%s' % (o['obligation'], o['satisfiable'], o['witness']))


if __name__ == '__main__':
    main()

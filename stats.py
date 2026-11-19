# -*- coding: utf-8 -*-
"""Step 9: descriptive statistics, Mann-Whitney U with Vargha-Delaney A12 and Holm
correction, and one-sample Wilcoxon tests against the deterministic baseline B0."""
import json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

D = os.path.join(HERE, 'data', '')
CONFIGS = ['B1', 'B2', 'B3', "A1'", "A3'"]
METRICS = ['full_ms_S1_eval', 'full_ms_S1_eval_2nd', 'full_ms_S2', 'min_ms_S1_eval', 'min_ms_S2', 'minimised_size',
           'bcov', 'mcdc', 'obligations', 'full_ms_S1_train', 'seconds']
COMPARISONS = [('B3', 'B1'), ('B3', 'B2'), ('B2', 'B1'), ('B3', "A1'"), ('B3', "A3'"), ("A1'", 'B2'), ("A3'", 'B2')]


def a12(x, y):
    """Vargha-Delaney A12: probability that a value from x exceeds one from y (ties count half)."""
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum(); eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ev = json.load(open(D + 'evaluation.json'))
    runs = ev['runs']
    by = {c: [r for r in runs if r['config'] == c] for c in CONFIGS}
    b0 = next(r for r in runs if r['config'] == 'B0')
    desc = {}
    for c in CONFIGS + ['B0']:
        rr = by.get(c, [b0])
        desc[c] = {}
        for m in METRICS:
            v = np.array([r[m] for r in rr], float)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            desc[c][m] = dict(n=len(v), median=float(med), iqr=float(q3 - q1), min=float(v.min()), max=float(v.max()),
                              mean=float(v.mean()), all_equal_max=bool((v == v.max()).all()))
        desc[c]['seeds_with_all_12_obligations'] = int(sum(r['obligations'] == 12 for r in rr))
    tests = {}
    for m in METRICS:
        ps, rows = [], []
        for a, b in COMPARISONS:
            x = [r[m] for r in by[a]]; y = [r[m] for r in by[b]]
            if np.ptp(x + y) == 0:
                p = 1.0
            else:
                p = mannwhitneyu(x, y, alternative='two-sided').pvalue
            rows.append(dict(a=a, b=b, A12=float(a12(x, y)), p=float(p))); ps.append(p)
        for r, pa in zip(rows, holm(np.array(ps))):
            r['p_holm'] = float(pa)
        tests[m] = rows
    vs_b0 = {}
    for m in METRICS:
        if m == 'seconds':
            continue
        vs_b0[m] = {}
        for c in CONFIGS:
            d = np.array([r[m] for r in by[c]], float) - b0[m]
            p = 1.0 if (d == 0).all() else float(wilcoxon(d, zero_method='zsplit').pvalue)
            vs_b0[m][c] = dict(b0=b0[m], median_diff=float(np.median(d)), p=p,
                               better=int((d > 0).sum()), equal=int((d == 0).sum()), worse=int((d < 0).sum()))
    json.dump(dict(descriptive=desc, mann_whitney=tests, vs_B0=vs_b0), open(D + 'stats.json', 'w'), indent=1)
    for m in METRICS:
        print('\n' + m)
        for c in CONFIGS + ['B0']:
            d = desc[c][m]
            print('  %-4s median %.4f IQR %.4f [%.4f, %.4f]' % (c, d['median'], d['iqr'], d['min'], d['max']))
        for t in tests[m]:
            print('  %-4s vs %-4s A12 %.3f p %.3g p_holm %.3g' % (t['a'], t['b'], t['A12'], t['p'], t['p_holm']))
    print('\nseeds with all 12 obligations:', {c: desc[c]['seeds_with_all_12_obligations'] for c in CONFIGS + ['B0']})


if __name__ == '__main__':
    main()

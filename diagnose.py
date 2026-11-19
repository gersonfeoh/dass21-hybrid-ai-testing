# -*- coding: utf-8 -*-
"""Step 8 of the protocol: fault-family diagnosis from failure signatures.

Probe suite (fixed): the 35-vector oracle (B0) plus the exhaustive single-factor suite of
Article 1 (every lattice score of one subscale, the other two at 0, rotating construction).
Features use observed behaviour only: per probe vector the five output differences, the sign
of each category difference, failure counts per output, and the index of the first failing
probe vector. Mutant names, operators and parameters are never features.

Evaluation: StratifiedGroupKFold (k = 5, groups = fault sites) with nested tuning (inner
StratifiedGroupKFold, k = 3), and transfer S1 -> S2 on code-level mutants mapped to the
same families. Baselines: majority class, hand-written rules, logistic regression.
"""
import sys, json, itertools, warnings
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'article1'))
from s1 import Pool, ref_out, SUB, BUTIR, BATAS
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
D = os.path.join(HERE, 'data', '')


def probe_suite():
    import uji
    ex = uji.bangun(uji.satu_faktor(uji.suite_tuntas()), 'putar')
    B0 = np.load(D + 'B0.npy').astype(int).tolist()
    seen, out = set(), []
    for x in B0 + ex:
        if tuple(x) not in seen:
            seen.add(tuple(x)); out.append(x)
    return np.array(out, np.int8), len(B0)


def features(O, R):
    """O: (M,n,5) mutant outputs, R: (n,5) reference outputs -> feature matrix (M, p)."""
    diff = (O != R[None])
    sign = np.sign(O[:, :, :3].astype(int) - R[None, :, :3].astype(int))
    sign[(O[:, :, :3] < 0) | (O[:, :, :3] > 4)] = 2          # crash or unknown label
    counts = diff.sum(1)
    anyd = diff.any(2)
    first = np.where(anyd.any(1), anyd.argmax(1), -1)[:, None]
    return np.concatenate([diff.reshape(len(O), -1), sign.reshape(len(O), -1), counts, first], 1).astype(float)


def rules(O, R, n_b0, ex_meta):
    """Hand-written diagnosis from the failure signature (authors' baseline)."""
    pred = []
    for o in O:
        d = o != R
        cat = d[:, :3].any(1)
        if not d.any():
            pred.append('threshold'); continue                        # undetected: most frequent family
        if (o < 0).any() or (o[:, :3] > 4).any():
            pred.append('other'); continue
        if not cat.any():
            g, f = d[:, 3].any(), d[:, 4].any()
            pred.append('gate' if g and not f else 'flag' if f and not g else 'mixed'); continue
        # single-factor part: vector varies subscale s, others at 0
        cross, own_pts = False, {s: [] for s in range(3)}
        for k, (s, v) in enumerate(ex_meta):
            row = d[n_b0 + k]
            for t in range(3):
                if row[t]:
                    if t != s and v > 0:
                        cross = True
                    elif t == s:
                        own_pts[s].append(v)
        if cross:
            pred.append('mapping'); continue
        spans = [p for p in own_pts.values() if p]
        if any(len(p) >= 4 for p in spans) or len(spans) > 1:
            pred.append('scaling'); continue
        # gate or flag differences that the category differences do not explain point to a second fault
        gate_only = (d[:, 3] & ~cat).any() or (d[:, 4] & ~cat).any()
        pred.append('mixed' if gate_only else 'threshold')
    return np.array(pred)


def metrics(y, p, labels):
    return dict(accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, labels=labels, average='macro', zero_division=0),
                balanced_accuracy=balanced_accuracy_score(y, p),
                precision=precision_score(y, p, labels=labels, average='macro', zero_division=0),
                recall=recall_score(y, p, labels=labels, average='macro', zero_division=0))


def boot_ci(y, p, labels, n=10000, seed=2026):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v.append(f1_score(y[i], p[i], labels=labels, average='macro', zero_division=0))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mcnemar(y, a, b):
    from scipy.stats import binomtest
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(a_only_correct=n01, b_only_correct=n10, p=float(p))


def models(seed):
    return {
        'Logistic': (LogisticRegression(max_iter=5000, class_weight='balanced'), {'C': [0.1, 1, 10]}),
        'RF': (RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=1),
               {'max_depth': [None, 8], 'min_samples_leaf': [1, 3]}),
        'XGBoost': (XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=seed, n_jobs=1, tree_method='hist',
                                  eval_metric='mlogloss'), {'max_depth': [3, 6], 'subsample': [0.8, 1.0]}),
    }


def main():
    m = json.load(open(D + 'mutants_S1.json'))
    muts = [x for x in m['first_order'] + m['second_order'] if not x['equivalent']]
    P, n_b0 = probe_suite()
    import uji
    ex_meta = [(SUB.index(s), v) for s in SUB for v in uji.KISI]
    # keep only metadata of the exhaustive vectors that survived de-duplication
    ex_vecs = uji.bangun(uji.satu_faktor(uji.suite_tuntas()), 'putar')
    keep, seen = [], set(map(tuple, np.load(D + 'B0.npy').astype(int).tolist()))
    for x, meta in zip(ex_vecs, ex_meta):
        if tuple(x) not in seen:
            seen.add(tuple(x)); keep.append(meta)
    ex_meta = keep
    R = ref_out(P)
    O = Pool([x['params'] for x in muts]).run(P)
    X = features(O, R)
    y = np.array([x['family'] for x in muts])
    # group = fault site; a second-order mutant is grouped with the first site of its pair
    groups = np.array([x['site'].split('|')[0] for x in muts])
    det = (O != R[None]).any((1, 2))
    labels = sorted(str(c) for c in set(y))
    print('S1 dataset: %d non-equivalent mutants, %d features, probe suite %d vectors, detected by probe %d' %
          (len(y), X.shape[1], len(P), det.sum()))
    print('class counts:', {c: int((y == c).sum()) for c in labels}, '| sites per class:',
          {c: len(set(groups[y == c])) for c in labels})
    le = LabelEncoder().fit(labels)
    yi = le.transform(y)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    preds = {k: np.empty(len(y), dtype=object) for k in ('Majority', 'Rules', 'Logistic', 'RF', 'XGBoost')}
    preds['Rules'][:] = rules(O, R, n_b0, ex_meta)
    best_params = {k: [] for k in ('Logistic', 'RF', 'XGBoost')}
    for fold, (tr, te) in enumerate(outer.split(X, yi, groups)):
        preds['Majority'][te] = DummyClassifier(strategy='most_frequent').fit(X[tr], y[tr]).predict(X[te])
        for name, (est, grid) in models(2026 + fold).items():
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=fold)
            gs = GridSearchCV(est, grid, cv=inner.split(X[tr], yi[tr], groups[tr]), scoring='f1_macro', n_jobs=1)
            sw = compute_sample_weight('balanced', yi[tr]) if name == 'XGBoost' else None
            if sw is not None:
                gs.fit(X[tr], yi[tr], sample_weight=sw)
            else:
                gs.fit(X[tr], yi[tr])
            preds[name][te] = le.inverse_transform(gs.predict(X[te]))
            best_params[name].append(gs.best_params_)
    res = {'S1_groupkfold': {}}
    for k, p in preds.items():
        p = p.astype(str)
        res['S1_groupkfold'][k] = dict(metrics(y, p, labels), macro_f1_ci95=boot_ci(y, p, labels),
                                       per_class_f1=dict(zip(labels, f1_score(y, p, labels=labels, average=None, zero_division=0).tolist())),
                                       confusion=confusion_matrix(y, p, labels=labels).tolist())
    res['labels'] = labels
    res['mcnemar'] = {'RF vs Rules': mcnemar(y, preds['RF'].astype(str), preds['Rules'].astype(str)),
                      'XGBoost vs Rules': mcnemar(y, preds['XGBoost'].astype(str), preds['Rules'].astype(str)),
                      'RF vs XGBoost': mcnemar(y, preds['RF'].astype(str), preds['XGBoost'].astype(str)),
                      'RF vs Logistic': mcnemar(y, preds['RF'].astype(str), preds['Logistic'].astype(str)),
                      'XGBoost vs Majority': mcnemar(y, preds['XGBoost'].astype(str), preds['Majority'].astype(str)),
                      'RF vs Majority': mcnemar(y, preds['RF'].astype(str), preds['Majority'].astype(str))}
    res['best_params'] = best_params
    res['dataset'] = dict(n=len(y), n_features=X.shape[1], probe_vectors=len(P), detected_by_probe=int(det.sum()),
                          class_counts={c: int((y == c).sum()) for c in labels},
                          sites_per_class={c: len(set(groups[y == c])) for c in labels})
    # rules on first-order only (is the rule baseline already perfect there?)
    fo = np.array([x['order'] == 1 for x in muts])
    res['rules_first_order_only'] = metrics(y[fo], preds['Rules'].astype(str)[fo], sorted(set(y[fo])))
    res['per_order'] = {k: {'first_order': metrics(y[fo], preds[k].astype(str)[fo], sorted(set(y[fo]))),
                            'second_order': metrics(y[~fo], preds[k].astype(str)[~fo], sorted(set(y[~fo])))} for k in preds}

    # transfer S1 -> S2
    try:
        s2 = json.load(open(D + 's2_labels.json'))
    except FileNotFoundError:
        s2 = None
    from evaluate_lib import s2_available
    if s2 and s2_available():
        import subprocess
        P.tofile(D + 'probe.bin')
        subprocess.run(['node', os.path.join(HERE, 's2', 'eval_mutants.js'), os.path.join(HERE, 's2', 'reports', 'mutation.json'),
                        D + 'probe.bin', D + 'probe_s2.out'], check=True, capture_output=True)
        O2 = np.fromfile(D + 'probe_s2.out', np.int8).reshape(-1, len(P), 5)
        assert (O2[0] == R).all(), 'S2 original differs from the reference on the probe suite'
        ids = json.load(open(D + 'probe_s2.out.ids.json'))
        lab = {str(r['id']): r for r in s2}
        sel = [k for k, mm in enumerate(ids) if lab[str(mm['id'])]['family'] in labels and not lab[str(mm['id'])]['equivalent']]
        X2 = features(O2[1:][sel], R); y2 = np.array([lab[str(ids[k]['id'])]['family'] for k in sel])
        res['S1_to_S2'] = {}
        p_rules = rules(O2[1:][sel], R, n_b0, ex_meta)
        labels2 = sorted(set(y2))
        res['S1_to_S2']['Rules'] = metrics(y2, p_rules, labels2)
        res['S1_to_S2']['Majority'] = metrics(y2, np.array([DummyClassifier(strategy='most_frequent').fit(X, y).predict(X2[:1])[0]] * len(y2)), labels2)
        for name, (est, grid) in models(2026).items():
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
            gs = GridSearchCV(est, grid, cv=inner.split(X, yi, groups), scoring='f1_macro', n_jobs=1)
            if name == 'XGBoost':
                gs.fit(X, yi, sample_weight=compute_sample_weight('balanced', yi))
            else:
                gs.fit(X, yi)
            p2 = le.inverse_transform(gs.predict(X2))
            res['S1_to_S2'][name] = dict(metrics(y2, p2, labels2), confusion=confusion_matrix(y2, p2, labels=labels2).tolist(),
                                         per_class_f1=dict(zip(labels2, f1_score(y2, p2, labels=labels2, average=None, zero_division=0).tolist())))
        res['S1_to_S2_labels'] = labels2
        res['S1_to_S2_dataset'] = dict(n=len(y2), class_counts={c: int((y2 == c).sum()) for c in sorted(set(y2))})
    json.dump(res, open(D + 'diagnosis_results.json', 'w'), indent=1, default=str)
    print('%-9s %8s %8s %8s %8s %8s  %s' % ('model', 'acc', 'macroF1', 'balacc', 'prec', 'rec', 'F1 95% CI'))
    for k, v in res['S1_groupkfold'].items():
        print('%-9s %8.3f %8.3f %8.3f %8.3f %8.3f  %s' % (k, v['accuracy'], v['macro_f1'], v['balanced_accuracy'], v['precision'], v['recall'], np.round(v['macro_f1_ci95'], 3)))
    print('rules on first-order only:', {k: round(v, 3) for k, v in res['rules_first_order_only'].items()})
    print('McNemar:', res['mcnemar'])
    if 'S1_to_S2' in res:
        print('S1 -> S2:', res['S1_to_S2_dataset'])
        for k, v in res['S1_to_S2'].items():
            print('  %-9s %s' % (k, {a: round(b, 3) for a, b in v.items() if isinstance(b, float)}))


if __name__ == '__main__':
    main()

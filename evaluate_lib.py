# -*- coding: utf-8 -*-
"""Suite measurement shared by evaluate_runs.py and llm_rag/llm_experiment.py.

S1 pools are always measured. S2 (StrykerJS mutants of the platform code) is measured only
when the private files s2/src/dass21.js and s2/reports/mutation.json are present; they are
not distributed because they contain production source code."""
import json, os, subprocess
import numpy as np
from s1 import Pool, ref_out
from ga import Features, load_pools

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', '')
S2 = os.path.join(HERE, 's2', '')


def s2_available():
    if os.environ.get('DASS21_NO_S2'):
        return False
    return os.path.exists(S2 + 'reports/mutation.json') and os.path.exists(S2 + 'src/dass21.js')


def minimise(rows):
    """Greedy set cover: keep a subset of vectors whose union of feature rows equals the
    union of the whole suite (ties broken by original order)."""
    target = rows.any(0)
    covered = np.zeros_like(target); keep = []
    while (covered != target).any():
        gain = (rows & ~covered).sum(1)
        i = int(np.argmax(gain)); keep.append(i); covered |= rows[i]
    return sorted(keep)


class Evaluator:
    def __init__(self, with_s2=None, tag='suites_all'):
        self.with_s2 = s2_available() if with_s2 is None else with_s2
        self.tag = tag
        m, train = load_pools()
        fo, so = m['first_order'], m['second_order']
        self.pools = {
            'S1_eval': [x for x in fo if x['pool'] == 'eval'],
            'S1_eval_2nd': [x for x in so if x['pool'] == 'eval'],
            'S1_all_1st': [x for x in fo if not x['equivalent']],
            'S1_train': train,
        }
        self.P = {k: Pool([x['params'] for x in v]) for k, v in self.pools.items()}
        self.feats = Features(train)
        self.s2lab = {str(r['id']): r for r in json.load(open(D + 's2_labels.json'))}

    def kill_matrices(self, allv):
        R = ref_out(allv)
        kills = {k: (self.P[k].run(allv) != R[None]).any(2).T for k in self.P}
        if self.with_s2:
            allv.tofile(D + self.tag + '.bin')
            subprocess.run(['node', S2 + 'eval_mutants.js', S2 + 'reports/mutation.json', D + self.tag + '.bin', D + self.tag + '_s2.out'],
                           check=True, capture_output=True)
            O2 = np.fromfile(D + self.tag + '_s2.out', np.int8).reshape(-1, len(allv), 5)
            assert (O2[0] == R).all(), 'S2 original differs from the reference engine'
            ids = json.load(open(D + self.tag + '_s2.out.ids.json'))
            ne = np.array([not self.s2lab[str(i['id'])]['equivalent'] for i in ids])
            kills['S2'] = (O2[1:][ne] != R[None]).any(2).T
            self.s2_ids = [str(i['id']) for i, keep in zip(ids, ne) if keep]
        return kills

    def evaluate(self, runs):
        allv = np.unique(np.concatenate([np.array(r['suite'], np.int8).reshape(-1, 21) for r in runs]), axis=0)
        kills = self.kill_matrices(allv)
        index = {v.tobytes(): i for i, v in enumerate(allv)}
        Mt = self.feats.Mt
        out = []
        for r in runs:
            S = np.array(r['suite'], np.int8).reshape(-1, 21)
            idx = np.array([index[v.tobytes()] for v in S])
            rows = self.feats.rows[self.feats.rows_for(S)]
            keep = minimise(rows)
            res = dict(config=r['config'], seed=r['seed'], seconds=r.get('seconds', 0.0), n_eval=r.get('n_eval', 0),
                       unique=len(set(map(bytes, S))), minimised_size=len(keep),
                       bcov=float(rows[:, Mt:Mt + 24].any(0).mean()), mcdc=float(rows[:, Mt + 24:Mt + 33].any(0).mean()),
                       obligations=int(rows[:, Mt + 33:].any(0).sum()))
            if 'n_fixed_llm' in r:
                res['n_fixed_llm'] = r['n_fixed_llm']
            for name, sub in (('full', idx), ('min', idx[keep])):
                for k in kills:
                    res['%s_ms_%s' % (name, k)] = float(kills[k][sub].any(0).mean())
            out.append(res)
        return out, kills, index

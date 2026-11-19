# -*- coding: utf-8 -*-
"""Steps 5 and 6 of the protocol: test-suite generation.

Configurations (Article 2 manuscript, Table 5):
  B1  random search, raw genotype, equal budget of 2 x 10^4 suite evaluations
  B2  NSGA-II, raw genotype, no formal constraints
  B3  NSGA-II, lattice-aware genotype, Z3 safety obligations as hard constraints
  A1' B3 without Z3 constraints (lattice-aware genotype only)
  A3' B3 with the raw genotype (constraints only)
With validated LLM-RAG tests (see llm_rag/llm_experiment.py) the integrated configurations are
  B5  B3 with the validated LLM-RAG tests of the same seed fixed inside the suite
  A1  B5 without Z3 constraints, A3 B5 with the raw genotype; A2 (B5 without LLM seeds) is B3.

Objectives (all minimised by pymoo):
  -MS on the S1 training pool, -lattice boundary coverage (24 points),
  -unique-cause MC/DC coverage for G and F (9 condition vectors), number of unique vectors.
Constraint (B3, A3'): number of unmet Z3 safety obligations <= 0 (Deb's constrained domination).
"""
import json, time, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from s1 import SUB, BUTIR, BATAS, KRITIS, Pool, ref_out
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

D = os.path.join(HERE, 'data', '')
B = 35                      # suite size, fixed for a like-for-like comparison with B0
BUDGET = 20000              # suite evaluations per run
IDX = {s: np.array(BUTIR[s]) - 1 for s in SUB}
POS = np.zeros((3, 7), int)
for si, s in enumerate(SUB):
    POS[si] = IDX[s]

# boundary points L(c), U(c) for every published cut-point
LAT = np.arange(0, 43, 2)
BPOINTS = []
for si, s in enumerate(SUB):
    for c in BATAS[s]:
        BPOINTS.append((si, LAT[LAT < c].max()))
        BPOINTS.append((si, LAT[LAT >= c].min()))
BP_S = np.array([b[0] for b in BPOINTS]); BP_V = np.array([b[1] for b in BPOINTS])


def load_pools():
    m = json.load(open(D + 'mutants_S1.json'))
    fo = m['first_order']; so = m['second_order']
    train = [x for x in fo if x['pool'] == 'train']
    return m, train


class Features:
    """Per-vector feature rows with a cache: kills on the training pool, boundary points,
    MC/DC condition vectors, and safety obligations."""

    def __init__(self, train):
        self.pool = Pool([t['params'] for t in train])
        self.Mt = len(train)
        self.cache = {}
        self.rows = np.zeros((1 << 16, self.Mt + 24 + 9 + 12), bool)
        self.n = 0

    @staticmethod
    def describe(X):
        X = np.asarray(X, np.int16)
        ref = ref_out(X)
        sc = np.stack([2 * X[:, IDX[s]].sum(1) for s in SUB], 1)
        bp = sc[:, BP_S] == BP_V[None, :]
        N = ref[:, :3] == 0
        C = ref[:, 0] >= 3
        I = X[:, np.array(KRITIS) - 1] == 3
        nI = I.sum(1)
        G = ref[:, 3] == 1
        mc = np.stack([N.all(1), ~N[:, 0] & N[:, 1] & N[:, 2], N[:, 0] & ~N[:, 1] & N[:, 2], N[:, 0] & N[:, 1] & ~N[:, 2],
                       ~C & (nI == 0), C & (nI == 0), ~C & I[:, 0] & (nI == 1), ~C & I[:, 1] & (nI == 1), ~C & I[:, 2] & (nI == 1)], 1)
        p7 = np.stack([G & I[:, k] & (nI == 1) for k in range(3)], 1)
        ob = np.concatenate([mc, p7], 1)
        return ref, bp, mc, ob

    def rows_for(self, X):
        """X: (n,21) int8 -> row indices into self.rows (computing missing rows)."""
        keys = [x.tobytes() for x in X]
        idx = np.empty(len(keys), np.int64)
        new = {}
        for k, key in enumerate(keys):
            j = self.cache.get(key)
            if j is None:
                j = new.get(key)
                if j is None:
                    j = self.n + len(new); new[key] = j
            idx[k] = j
        if new:
            Xn = np.frombuffer(b''.join(new.keys()), np.int8).reshape(-1, 21)
            while self.n + len(new) > self.rows.shape[0]:
                self.rows = np.concatenate([self.rows, np.zeros_like(self.rows)], 0)
            ref, bp, mc, ob = self.describe(Xn)
            kill = (self.pool.run(Xn) != ref[None]).any(-1).T
            self.rows[self.n:self.n + len(new)] = np.concatenate([kill, bp, mc, ob], 1)
            self.cache.update(new); self.n += len(new)
        return idx

    def score(self, suites):
        """suites: (P,B,21) -> metrics (P,5): MS_train, bcov, mcdc, unique, obligations met."""
        P = suites.shape[0]
        idx = self.rows_for(suites.reshape(-1, 21)).reshape(P, suites.shape[1])
        cov = self.rows[idx].any(1)
        Mt = self.Mt
        uniq = np.array([len(set(r.tolist())) for r in idx])
        return np.stack([cov[:, :Mt].mean(1), cov[:, Mt:Mt + 24].mean(1), cov[:, Mt + 24:Mt + 33].mean(1),
                         uniq, cov[:, Mt + 33:].sum(1)], 1)


# ------------------------------------------------------------------ genotypes
class Raw:
    W = 21
    xu = 3

    @staticmethod
    def decode(Xpop):
        return Xpop.reshape(-1, B, 21).astype(np.int8)

    @staticmethod
    def random(rng, n):
        return rng.integers(0, 4, (n, B * 21))


class Lattice:
    """Per vector and subscale: raw sum r in 0..21 (score 2r on the lattice) and a
    permutation of the 7 item positions (filling order)."""
    W = 24
    xu = 21

    @staticmethod
    def random(rng, n):
        X = np.zeros((n, B, 24), int)
        X[:, :, :3] = rng.integers(0, 22, (n, B, 3))
        X[:, :, 3:] = np.argsort(rng.random((n, B, 3, 7)), -1).reshape(n, B, 21)
        return X.reshape(n, -1)

    @staticmethod
    def decode(Xpop):
        G = Xpop.reshape(-1, B, 24)
        n = G.shape[0]
        out = np.zeros((n, B, 21), np.int8)
        r = G[:, :, :3].copy()
        perm = G[:, :, 3:].reshape(n, B, 3, 7)
        for k in range(7):
            pos = perm[:, :, :, k]                                 # (n,B,3) position index 0..6
            item = POS[np.arange(3)[None, None, :], pos]           # (n,B,3) item index 0..20
            v = np.minimum(3, r)
            np.put_along_axis(out.reshape(n * B, 21), item.reshape(n * B, 3), v.reshape(n * B, 3), axis=1)
            r = r - v
        return out


# ------------------------------------------------------------------ operators
class GenSampling(Sampling):
    def __init__(self, geno):
        super().__init__(); self.geno = geno

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        return self.geno.random(random_state, n_samples)


class SuiteCrossover(Crossover):
    """Uniform crossover at vector level (p = 0.8) or one-point crossover inside one
    vector (p = 0.2); for the lattice genotype the cut lies between subscale blocks."""

    def __init__(self, geno):
        super().__init__(2, 2, prob=1.0); self.geno = geno

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = random_state
        _, n_m, n_var = X.shape
        w = n_var // B
        P1 = X[0].reshape(n_m, B, w).copy(); P2 = X[1].reshape(n_m, B, w).copy()
        C1, C2 = P1.copy(), P2.copy()
        for m in range(n_m):
            if rng.random() < 0.8:
                sw = rng.random(B) < 0.5
                C1[m, sw], C2[m, sw] = P2[m, sw], P1[m, sw]
            else:
                v = rng.integers(B)
                if self.geno is Raw:
                    cut = rng.integers(1, 21)
                    C1[m, v, cut:], C2[m, v, cut:] = P2[m, v, cut:], P1[m, v, cut:]
                else:
                    cut = rng.integers(1, 3)          # subscale block boundary
                    a = np.r_[cut:3, 3 + 7 * cut:24]
                    C1[m, v, a], C2[m, v, a] = P2[m, v, a], P1[m, v, a]
        return np.stack([C1.reshape(n_m, n_var), C2.reshape(n_m, n_var)])


class SuiteMutation(Mutation):
    """Raw: item reset with p = 1/21 per gene. Lattice: per vector and subscale with p = 1/6,
    move the score to a lattice neighbour or swap two filling positions. Both: replace a
    vector by a random one with p = 1/B."""

    def __init__(self, geno):
        super().__init__(prob=1.0); self.geno = geno

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = random_state
        n = X.shape[0]
        if self.geno is Raw:
            G = X.reshape(n, B, 21).copy()
            m = rng.random(G.shape) < 1 / 21
            G[m] = rng.integers(0, 4, m.sum())
            rep = rng.random((n, B)) < 1 / B
            G[rep] = rng.integers(0, 4, (rep.sum(), 21))
            return G.reshape(n, -1)
        G = X.reshape(n, B, 24).copy()
        for a, b, si in zip(*np.nonzero(rng.random((n, B, 3)) < 1 / 6)):
            if rng.random() < 0.5:
                G[a, b, si] = np.clip(G[a, b, si] + (1 if rng.random() < 0.5 else -1), 0, 21)
            else:
                i, j = rng.choice(7, 2, replace=False)
                o = 3 + 7 * si
                G[a, b, o + i], G[a, b, o + j] = G[a, b, o + j], G[a, b, o + i]
        rep = np.nonzero(rng.random((n, B)) < 1 / B)
        if len(rep[0]):
            G[rep] = Lattice.random(rng, len(rep[0])).reshape(-1, B, 24)[:, 0]
        return G.reshape(n, -1)


class SuiteProblem(Problem):
    def __init__(self, geno, feats, constrained, fixed=None):
        super().__init__(n_var=B * geno.W, n_obj=4, n_ieq_constr=1 if constrained else 0,
                         xl=0, xu=geno.xu, vtype=int)
        self.geno, self.feats, self.constrained, self.fixed = geno, feats, constrained, fixed

    def suites(self, X):
        S = self.geno.decode(np.asarray(X).astype(int))
        if self.fixed is not None and len(self.fixed):
            S = np.concatenate([S, np.broadcast_to(self.fixed, (S.shape[0],) + self.fixed.shape)], 1)
        return S

    def _evaluate(self, X, out, *args, **kwargs):
        m = self.feats.score(self.suites(X))
        out['F'] = np.stack([-m[:, 0], -m[:, 1], -m[:, 2], m[:, 3]], 1)
        if self.constrained:
            out['G'] = (12 - m[:, 4])[:, None].astype(float)


def pick(metrics, feasible):
    """Final suite: among feasible solutions, maximise MS_train, then boundary coverage,
    then MC/DC, then minimise unique vectors."""
    cand = np.nonzero(feasible)[0] if feasible.any() else np.argsort(-metrics[:, 4])[:1]
    key = sorted(cand, key=lambda i: (-metrics[i, 0], -metrics[i, 1], -metrics[i, 2], metrics[i, 3]))
    return key[0]


CONFIGS = {
    'B1': dict(search='random', geno=Raw, constrained=False),
    'B2': dict(search='nsga2', geno=Raw, constrained=False),
    'B3': dict(search='nsga2', geno=Lattice, constrained=True),
    "A1'": dict(search='nsga2', geno=Lattice, constrained=False),
    "A3'": dict(search='nsga2', geno=Raw, constrained=True),
    'B5': dict(search='nsga2', geno=Lattice, constrained=True, llm=True),
    'A1': dict(search='nsga2', geno=Lattice, constrained=False, llm=True),
    'A3': dict(search='nsga2', geno=Raw, constrained=True, llm=True),
}


def run(config, seed, pop_size=100, train=None, fixed=None):
    """fixed: validated LLM-RAG vectors kept inside every suite (B5, A1, A3); the GA then
    evolves the remaining B - len(fixed) vectors."""
    global B
    cfg = CONFIGS[config]
    fixed = None if fixed is None or not len(fixed) else np.unique(np.asarray(fixed, np.int8), axis=0)
    if cfg.get('llm') and fixed is None:
        raise ValueError('configuration %s needs validated LLM-RAG vectors' % config)
    B_full = B
    if fixed is not None:
        B = B_full - len(fixed)
    try:
        return _run(cfg, config, seed, pop_size, train, fixed)
    finally:
        B = B_full


def _run(cfg, config, seed, pop_size, train, fixed):
    if train is None:
        _, train = load_pools()
    feats = Features(train)
    t0 = time.time()
    if cfg['search'] == 'random':
        rng = np.random.default_rng(seed)
        best, best_m = None, None
        for _ in range(BUDGET // 500):
            S = Raw.decode(Raw.random(rng, 500))
            m = feats.score(S)
            i = pick(m, np.ones(len(m), bool))
            if best is None or tuple([-m[i, 0], -m[i, 1], -m[i, 2], m[i, 3]]) < tuple([-best_m[0], -best_m[1], -best_m[2], best_m[3]]):
                best, best_m = S[i].copy(), m[i].copy()
        suite, n_eval = best, BUDGET
    else:
        geno = cfg['geno']
        prob = SuiteProblem(geno, feats, cfg['constrained'], fixed)
        alg = NSGA2(pop_size=pop_size, sampling=GenSampling(geno), crossover=SuiteCrossover(geno),
                    mutation=SuiteMutation(geno), eliminate_duplicates=False)
        res = minimize(prob, alg, ('n_eval', BUDGET), seed=seed, verbose=False)
        pop = res.pop
        X = pop.get('X').astype(int)
        S = prob.suites(X)
        m = feats.score(S)
        feas = (12 - m[:, 4] <= 0) if cfg['constrained'] else np.ones(len(m), bool)
        i = pick(m, feas)
        suite, n_eval = S[i], res.algorithm.evaluator.n_eval
    dt = time.time() - t0
    m = feats.score(suite[None])[0]
    return dict(config=config, seed=seed, pop_size=pop_size, n_eval=int(n_eval), seconds=round(dt, 2),
                n_fixed_llm=0 if fixed is None else int(len(fixed)),
                train_ms=float(m[0]), bcov=float(m[1]), mcdc=float(m[2]), unique=int(m[3]), obligations=int(m[4]),
                suite=suite.astype(int).tolist(), distinct_vectors_evaluated=feats.n)


if __name__ == '__main__':
    r = run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 100)
    print({k: v for k, v in r.items() if k != 'suite'})

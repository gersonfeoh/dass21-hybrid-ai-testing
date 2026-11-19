# -*- coding: utf-8 -*-
"""Step 6 (measurement): evaluate the final suite of every run, and B0, on the held-out
S1 pools and on the S2 code-level mutants; measure coverage, safety obligations, size,
and the size after greedy minimisation (the same procedure for every configuration)."""
import json, glob, os
import numpy as np
from evaluate_lib import Evaluator, D, HERE


def main():
    ev = Evaluator(tag='suites_all')
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, 'runs', '*.json')))]
    B0 = np.load(D + 'B0.npy').astype(np.int8)
    runs.append(dict(config='B0', seed=0, suite=B0.tolist(), seconds=0.0, n_eval=0))
    out, kills, index = ev.evaluate(runs)
    for r in out:  # S2 key name kept as in the original log
        if 'full_ms_S2' not in r and ev.with_s2:
            raise RuntimeError('S2 missing')
    json.dump(dict(runs=out, pool_sizes={k: len(v) for k, v in ev.pools.items()},
                   s2_nonequivalent=len(ev.s2_ids) if ev.with_s2 else None,
                   s2_total=len(ev.s2lab), vectors=len(index)), open(D + 'evaluation.json', 'w'), indent=0)
    print('evaluated', len(out), 'suites;', {k: len(v) for k, v in ev.pools.items()}, 'S2 included', ev.with_s2)


if __name__ == '__main__':
    main()

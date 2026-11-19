# -*- coding: utf-8 -*-
"""Step 6: main runs, seeds 1 to 30 for every runnable stochastic configuration,
with the population size locked after the pilot (params_locked.json)."""
import json, os, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
from multiprocessing import Pool as MP
import ga

LOCK = json.load(open(os.path.join(HERE, 'pilot', 'params_locked.json')))
OUT = os.path.join(HERE, 'runs', '')


def job(arg):
    c, seed = arg
    path = OUT + '%s_%02d.json' % (c.replace("'", 'p'), seed)
    if os.path.exists(path):
        return path
    r = ga.run(c, seed, LOCK['pop_size'])
    json.dump(r, open(path, 'w'))
    return path


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jobs = [(c, s) for s in range(1, 31) for c in ('B1', 'B2', 'B3', "A1'", "A3'")]
    with MP(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        for k, path in enumerate(p.imap_unordered(job, jobs), 1):
            if k % 10 == 0:
                print('done', k, 'of', len(jobs), flush=True)
    print('all runs finished', flush=True)

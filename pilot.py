# -*- coding: utf-8 -*-
"""Step 5: pilot on the S1 training pool only (seeds 101 to 110) to lock the population size."""
import json, numpy as np, ga
_, train = ga.load_pools()
res = []
for pop in (50, 100):
    for c in ('B2', 'B3', "A1'", "A3'"):
        for seed in range(101, 111):
            r = ga.run(c, seed, pop, train); r.pop('suite'); res.append(r)
json.dump(res, open('pilot/pilot_log.json', 'w'), indent=0)
summ = {}
for pop in (50, 100):
    for c in ('B2', 'B3', "A1'", "A3'"):
        rr = [r for r in res if r['pop_size'] == pop and r['config'] == c]
        summ['%s pop %d' % (c, pop)] = {k: float(np.median([r[k] for r in rr])) for k in ('train_ms', 'bcov', 'mcdc', 'obligations', 'seconds')}
for k, v in summ.items():
    print(k, v)
json.dump(summ, open('pilot/pilot_summary.json', 'w'), indent=1)

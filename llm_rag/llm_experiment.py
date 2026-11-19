# -*- coding: utf-8 -*-
"""Protocol step 7: LLM-RAG experiment and the integrated configurations.

  Pools for Table 11 : temperature 0 (one sample per target, seed 0) and temperature 0.7
                       (ten samples per target, seeds 1 to 10).
  B4  seed s         : validated LLM-RAG tests of seed s at temperature 0.7 (no GA).
  B5  seed s         : B3 (lattice genotype + Z3 obligations) with the B4 tests of seed s fixed inside the suite.
  A1  seed s         : B5 without Z3 constraints.   A3 seed s: B5 with the raw genotype.
  A2                 : B5 without LLM seeds, which is B3 (runs/B3_*.json).

Usage (Ollama running locally, e.g. in Google Colab):
  python3 llm_rag/llm_experiment.py --model qwen2.5:7b-instruct --embed-model bge-m3 --seeds 30
Outputs go to results_llm/: llm_log.jsonl, llm_metrics.json, runs/*.json, evaluation_llm.json, stats_llm.json.
"""
import argparse, json, os, sys, time, glob
from multiprocessing import Pool as MP
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import pipeline as PL


def generate(a, out):
    ret = PL.Retriever(a.base_url, a.embed_model if a.embed_model != 'none' else None)
    prompts = [PL.build_prompt(ret, t) for t in PL.TARGETS]
    log_path = os.path.join(out, 'llm_log.jsonl')
    done = set()
    if os.path.exists(log_path):                      # resume after a disconnected session
        for l in open(log_path):
            r = json.loads(l); done.add((r['temperature'], r['seed'], r['target_index']))
    calls = [(0.0, 0)] + [(0.7, s) for s in range(1, max(a.seeds, a.pool_samples) + 1)]
    with open(log_path, 'a', encoding='utf-8') as f:
        for temp, seed in calls:
            for ti, (t, (prompt, rids)) in enumerate(zip(PL.TARGETS, prompts)):
                if (temp, seed, ti) in done:
                    continue
                try:
                    text, usage, dt = PL.chat(a.base_url, a.model, prompt, temp, seed)
                except Exception as e:                # recorded, never silently dropped
                    text, usage, dt = '', dict(prompt_tokens=0, output_tokens=0, error=str(e)[:200]), 0.0
                f.write(json.dumps(dict(model=a.model, embed_model=a.embed_model, temperature=temp, seed=seed, target_index=ti,
                                        requirement=t[0], target=t[1], retrieved=rids, prompt=prompt, response=text,
                                        usage=usage, seconds=round(dt, 2)), ensure_ascii=False) + '\n')
                f.flush()
            print('generated temperature %.1f seed %d' % (temp, seed), flush=True)
    return [json.loads(l) for l in open(log_path, encoding='utf-8')]


def validate_group(rows):
    """Validate one group of responses with a shared duplicate set (one pool or one seed)."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (r['seed'], r['target_index'])):
        rec, reason = PL.validate(r['response'], PL.TARGETS[r['target_index']], r['retrieved'], seen)
        out.append(dict(r, valid=reason is None, reject_reason=reason, record=rec))
    return out


NO_LLM_EQUIVALENT = {'B5': 'B3', 'A1': 'A1p', 'A3': 'A3p'}   # same configuration without LLM seeds


def ga_job(arg):
    config, seed, pop, fixed, outdir = arg
    import ga
    path = os.path.join(outdir, '%s_%02d.json' % (config, seed))
    if os.path.exists(path):
        return path
    if not fixed:   # no valid LLM test for this seed: the configuration reduces to its run without LLM seeds
        r = json.load(open(os.path.join(ROOT, 'runs', '%s_%02d.json' % (NO_LLM_EQUIVALENT[config], seed))))
        r['config'] = config; r['n_fixed_llm'] = 0; r['reused_from'] = NO_LLM_EQUIVALENT[config]
    else:
        r = ga.run(config, seed, pop, fixed=np.array(fixed, np.int8))
    json.dump(r, open(path, 'w'))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://localhost:11434')
    ap.add_argument('--model', required=True); ap.add_argument('--embed-model', default='bge-m3')
    ap.add_argument('--seeds', type=int, default=30); ap.add_argument('--pool-samples', type=int, default=10)
    ap.add_argument('--budget', type=int, default=20000); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results_llm'))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'runs'), exist_ok=True)
    import ga
    ga.BUDGET = a.budget
    lock = json.load(open(os.path.join(ROOT, 'pilot', 'params_locked.json')))
    meta = dict(model=a.model, embed_model=a.embed_model, seeds=a.seeds, pool_samples=a.pool_samples, budget=a.budget,
                pop_size=lock['pop_size'], started=time.strftime('%Y-%m-%d %H:%M:%S'))
    try:
        import urllib.request
        meta['ollama_models'] = json.loads(urllib.request.urlopen(a.base_url.rstrip('/') + '/api/tags', timeout=30).read())['models']
    except Exception as e:
        meta['ollama_models'] = 'unavailable: %s' % e

    # 1. generation and validation
    rows = generate(a, a.out)
    pools = {'T0': validate_group([r for r in rows if r['temperature'] == 0.0]),
             'T07': validate_group([r for r in rows if r['temperature'] == 0.7 and 1 <= r['seed'] <= a.pool_samples])}
    per_seed = {s: validate_group([r for r in rows if r['temperature'] == 0.7 and r['seed'] == s]) for s in range(1, a.seeds + 1)}

    # 2. Table 11 metrics (novel-kill measured after evaluation below)
    from evaluate_lib import Evaluator, D
    ev = Evaluator(tag='llm_all')
    B0 = np.load(D + 'B0.npy').astype(np.int8)
    metric = {}
    for name, rr in pools.items():
        val = [r for r in rr if r['valid']]
        n = len(rr)
        metric[name] = dict(responses=n, valid=len(val), validity_rate=len(val) / n if n else 0,
                            oracle_agreement_rate=float(np.mean([r['record']['oracle_agrees'] for r in val])) if val else None,
                            target_met_rate=float(np.mean([r['record']['target_met'] for r in val])) if val else None,
                            traceability_precision=float(np.mean([r['record']['traceable'] for r in val])) if val else None,
                            cites_target_rate=float(np.mean([r['record']['cites_target'] for r in val])) if val else None,
                            duplicate_rate=sum(r['reject_reason'] == 'duplicate' for r in rr) / n if n else 0,
                            reject_reasons={k: sum(r['reject_reason'] == k for r in rr) for k in ('no_json', 'invalid_json', 'schema_or_domain', 'duplicate')},
                            errors=sum('error' in r['usage'] for r in rr),
                            tokens_per_valid_test=(sum(r['usage'].get('prompt_tokens', 0) + r['usage'].get('output_tokens', 0) for r in rr) / len(val)) if val else None,
                            seconds_per_valid_test=(sum(r['seconds'] for r in rr) / len(val)) if val else None, cost_usd=0.0)

    # 3. suites: B4 (LLM only), B5, A1, A3 (LLM-seeded GA)
    b4 = {s: [r['record']['items'] for r in per_seed[s] if r['valid']] for s in per_seed}
    runs = [dict(config='B4', seed=s, suite=v if v else B0[:1].tolist(), seconds=0.0, n_eval=0, empty=not v) for s, v in b4.items()]
    jobs = [(c, s, lock['pop_size'], b4[s], os.path.join(a.out, 'runs')) for s in per_seed for c in ('B5', 'A1', 'A3')]
    with MP(a.workers) as p:
        for k, _ in enumerate(p.imap_unordered(ga_job, jobs), 1):
            if k % 10 == 0:
                print('GA runs', k, 'of', len(jobs), flush=True)
    runs += [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.out, 'runs', '*.json')))]
    ref_runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, 'runs', 'B3_*.json')))]
    for r in ref_runs:
        r = dict(r); r['config'] = 'A2 (= B3)'; runs.append(r)
    runs.append(dict(config='B0', seed=0, suite=B0.tolist(), seconds=0.0, n_eval=0))

    # 4. measurement (S2 only when the private module is present)
    res, kills, index = ev.evaluate(runs)
    for r, src in zip(res, runs):
        if src.get('empty'):      # no valid LLM test: an empty suite detects nothing
            r['empty_suite'] = True
            for k in list(r):
                if '_ms_' in k or k in ('bcov', 'mcdc', 'obligations', 'minimised_size', 'unique'):
                    r[k] = 0
    # novel-kill: valid tests that kill a held-out mutant missed by B0
    b0_idx = [index[v.tobytes()] for v in B0]
    for name, rr in pools.items():
        val = [r for r in rr if r['valid']]
        if not val:
            continue
        nk = 0
        for r in val:
            i = index.get(np.array(r['record']['items'], np.int8).tobytes())
            if i is None:
                continue
            for k in [k for k in kills if k in ('S1_eval', 'S1_eval_2nd', 'S2')]:
                if (kills[k][i] & ~kills[k][b0_idx].any(0)).any():
                    nk += 1; break
        metric[name]['novel_kill_rate'] = nk / len(val)
        metric[name]['novel_kill_pools'] = [k for k in kills if k in ('S1_eval', 'S1_eval_2nd', 'S2')]

    # 5. statistics
    from stats import a12, holm
    from scipy.stats import mannwhitneyu
    by = {}
    for r in res:
        by.setdefault(r['config'], []).append(r)
    ms_keys = [k for k in ('full_ms_S1_eval', 'full_ms_S2', 'minimised_size', 'obligations') if k in res[0]]
    tests = {}
    for m in ms_keys:
        rows_t, ps = [], []
        for x, y in (('B5', 'A1'), ('B5', 'A2 (= B3)'), ('B5', 'A3'), ('B4', 'A2 (= B3)')):
            if x not in by or y not in by:
                continue
            u = [r[m] for r in by[x]]; v = [r[m] for r in by[y]]
            p = 1.0 if np.ptp(u + v) == 0 else float(mannwhitneyu(u, v, alternative='two-sided').pvalue)
            rows_t.append(dict(a=x, b=y, A12=float(a12(u, v)), p=p)); ps.append(p)
        for t, q in zip(rows_t, holm(np.array(ps)) if ps else []):
            t['p_holm'] = float(q)
        tests[m] = rows_t
    desc = {c: {m: dict(median=float(np.median([r[m] for r in rr])), q1=float(np.percentile([r[m] for r in rr], 25)),
                        q3=float(np.percentile([r[m] for r in rr], 75)), min=float(min(r[m] for r in rr)), max=float(max(r[m] for r in rr)))
                for m in ms_keys + ['bcov', 'mcdc', 'seconds']} for c, rr in by.items()}
    for c, rr in by.items():
        desc[c]['n'] = len(rr); desc[c]['seeds_with_all_12_obligations'] = sum(r['obligations'] == 12 for r in rr)
        desc[c]['empty_suites'] = sum(r.get('empty_suite', False) for r in rr)
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); meta['s2_included'] = ev.with_s2
    json.dump(dict(meta=meta, metrics=metric), open(os.path.join(a.out, 'llm_metrics.json'), 'w'), indent=1)
    json.dump(dict(meta=meta, runs=res), open(os.path.join(a.out, 'evaluation_llm.json'), 'w'), indent=0)
    json.dump(dict(meta=meta, descriptive=desc, mann_whitney=tests), open(os.path.join(a.out, 'stats_llm.json'), 'w'), indent=1)
    json.dump(dict(pools=pools), open(os.path.join(a.out, 'llm_validated.json'), 'w'), indent=0, default=str)
    print(json.dumps(dict(metrics=metric, descriptive={c: {k: v for k, v in d.items() if k in ('n', 'seeds_with_all_12_obligations', 'empty_suites')} | {m: d[m]['median'] for m in ms_keys if m in d} for c, d in desc.items()}), indent=1))


if __name__ == '__main__':
    main()

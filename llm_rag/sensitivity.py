# -*- coding: utf-8 -*-
"""Post hoc sensitivity analysis (reported as a protocol deviation, not as the primary result).

The pre-specified validator rejects a response when any field fails the JSON schema. The prompt
did not state the types of expected.G, expected.F and target_condition explicitly, and the
expected block is only a metric (never an oracle). This script re-validates the logged responses
with a lenient validator that still requires a JSON object with 21 integer items in 0..3 and
non-duplicate vectors, but tolerates type deviations in expected and target_condition.
No LLM call is made; the input is results_llm/llm_log.jsonl.
"""
import json, os, re, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import pipeline as PL


def lenient(raw, target, retrieved, seen):
    m = re.search(r'\{.*\}', raw or '', re.S)
    if not m:
        return None, 'no_json'
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, 'invalid_json'
    x = obj.get('items') if isinstance(obj, dict) else None
    if not (isinstance(x, list) and len(x) == 21 and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 3 for v in x)):
        return None, 'items_domain'
    if tuple(x) in seen:
        return None, 'duplicate'
    cond, o = PL.conditions(x)
    exp = obj.get('expected') if isinstance(obj.get('expected'), dict) else {}
    def b(v):
        return {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'open': 1, 'closed': 0}.get(str(v).strip().lower(), v if isinstance(v, int) else None)
    pred = [exp.get('c_D'), exp.get('c_A'), exp.get('c_S'), b(exp.get('G')), b(exp.get('F'))]
    rids = obj.get('requirement_ids') if isinstance(obj.get('requirement_ids'), list) else []
    seen.add(tuple(x))
    return dict(items=x, oracle=o.tolist(), oracle_agrees=[int(v) if isinstance(v, (int, bool)) else v for v in pred] == o.tolist(),
                target_met=all(cond[k] == v for k, v in target[2].items()),
                traceable=bool(rids) and all(r in retrieved for r in rids), cites_target=target[0] in rids), None


def main(out):
    rows = [json.loads(l) for l in open(os.path.join(out, 'llm_log.jsonl'), encoding='utf-8')]
    pools = {'T0': [r for r in rows if r['temperature'] == 0.0], 'T07': [r for r in rows if r['temperature'] == 0.7 and 1 <= r['seed'] <= 10]}
    res = {}
    for name, rr in pools.items():
        for mode, fn in (('strict', PL.validate), ('lenient', lenient)):
            seen, recs, reasons = set(), [], Counter()
            for r in sorted(rr, key=lambda r: (r['seed'], r['target_index'])):
                rec, why = fn(r['response'], PL.TARGETS[r['target_index']], r['retrieved'], seen)
                reasons[why or 'valid'] += 1
                if rec:
                    recs.append(rec)
            n = len(rr)
            res['%s_%s' % (name, mode)] = dict(responses=n, valid=len(recs), validity_rate=len(recs) / n if n else 0,
                                               oracle_agreement_rate=sum(r['oracle_agrees'] for r in recs) / len(recs) if recs else None,
                                               target_met_rate=sum(r['target_met'] for r in recs) / len(recs) if recs else None,
                                               traceability_precision=sum(r['traceable'] for r in recs) / len(recs) if recs else None,
                                               reasons=dict(reasons))
    # schema error profile of the pre-specified validator
    import jsonschema
    prof = Counter()
    for r in rows:
        try:
            o = json.loads(re.search(r'\{.*\}', r['response'] or '', re.S).group(0))
        except Exception:
            prof['no or invalid JSON'] += 1; continue
        errs = list(jsonschema.Draft7Validator(PL.SCHEMA).iter_errors(o))
        for key in sorted({'/'.join(map(str, list(e.path)[:1])) + ':' + e.validator for e in errs}):
            prof[key] += 1
    res['schema_error_profile_all_responses'] = dict(prof.most_common())
    json.dump(res, open(os.path.join(out, 'sensitivity_lenient.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'results_llm'))

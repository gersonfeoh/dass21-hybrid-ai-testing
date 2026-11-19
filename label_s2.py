# -*- coding: utf-8 -*-
"""S2 mutant labels: fault family (single rater, rule by source location) and the
manual equivalence review of every mutant that survived the structured input set.

Review categories for survivors:
  equivalent      identical behaviour for every valid response vector {0..3}^21 with one screening
  not_observable  behaviour differs only in state the stateless oracle cannot see
                  (empty screening history) or in response fields outside the five outputs
"""
import json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', '')

REVIEW = {
    79: ('equivalent', 'loop bound i <= t.length is reached only when the score exceeds every upper bound; the last bound is 42'),
    86: ('equivalent', 'fallback after the loop is unreachable because every valid score is <= 42'),
    89: ('equivalent', 'length guard replaced by false; every valid input has 21 answers'),
    90: ('equivalent', '!jawaban && length !== 21 is false for every valid input, as is the original'),
    92: ('equivalent', 'length check replaced by false; every valid input has 21 answers'),
    94: ('equivalent', 'body of the guard is unreachable for valid input'),
    95: ('equivalent', 'body of the guard is unreachable for valid input'),
    96: ('equivalent', 'body of the guard is unreachable for valid input'),
    101: ('equivalent', 'sanitiser disabled; no valid answer needs sanitising'),
    102: ('equivalent', '(isNaN or v < 0) and v > 3 is false for every valid answer, as is the original'),
    103: ('equivalent', 'isNaN(v) or v < 0 replaced by false; both are false for valid answers'),
    104: ('equivalent', 'isNaN(v) and v < 0 is false for valid answers, as is the original'),
    105: ('equivalent', 'v < 0 replaced by false; false for valid answers'),
    106: ('equivalent', 'v <= 0 resets 0 to 0'),
    108: ('equivalent', 'v > 3 replaced by false; false for valid answers'),
    172: ('equivalent', 'non-Normal branch returns {} instead of {boleh: false}; boleh is then undefined, which the gate treats as not eligible'),
    149: ('not_observable', 'empty-history check removed; affects only students who never screened (stateful rule)'),
    150: ('not_observable', 'empty-history response emptied; stateful rule'),
    151: ('not_observable', 'students with no screening become eligible (boleh: true); stateful rule, safety relevant'),
    152: ('not_observable', 'reason string of the empty-history response; stateful rule'),
    174: ('not_observable', 'reason string of the non-Normal response is not part of the five outputs'),
    175: ('not_observable', 'category echo of the non-Normal response is not part of the five outputs'),
}


def family(line):
    if 9 <= line <= 31:
        return 'mapping'        # DASS_ITEMS: item-to-subscale table
    if 33 <= line <= 43:
        return 'threshold'      # DASS_THRESHOLDS and kategoriDass_
    if line in (46, 47, 52):
        return 'other'          # input guard and sanitiser (no S1 counterpart)
    if 49 <= line <= 55:
        return 'scaling'        # score accumulation and doubling
    if 57 <= line <= 60:
        return 'threshold'      # category assignment per subscale
    if 63 <= line <= 67:
        return 'flag'
    if line == 69:
        return 'other'          # result object
    if 73 <= line <= 83:
        return 'gate'
    return 'other'              # module exports


def main():
    ids = json.load(open(D + 'b0_s2.out.ids.json'))
    kills = {str(k['id']): k for k in json.load(open(D + 'equiv_s2_kill.json'))}
    survivors = {int(m['id']) for m in ids if kills[str(m['id'])]['kills'] == 0}
    reviewed = set(REVIEW)
    print('survivors of the structured input set:', sorted(survivors))
    print('reviewed but killed:', sorted(reviewed - survivors), '| surviving but not reviewed:', sorted(survivors - reviewed))
    out = []
    for m in ids:
        i = int(m['id'])
        rv = REVIEW.get(i) if i in survivors else None
        out.append(dict(id=m['id'], mutator=m['mutator'], line=m['line'], original=m['original'][:80], replacement=m['replacement'][:80],
                        family=family(m['line']), review=rv[0] if rv else 'killed', reason=rv[1] if rv else '',
                        equivalent=bool(rv), stryker_status_B0=m['stryker_status'],
                        kills_structured_set=kills[str(m['id'])]['kills']))
    json.dump(out, open(D + 's2_labels.json', 'w'), indent=1)
    from collections import Counter
    print('families:', Counter(o['family'] for o in out))
    print('review:', Counter(o['review'] for o in out))
    print('non-equivalent by family:', Counter(o['family'] for o in out if not o['equivalent']))


if __name__ == '__main__':
    main()

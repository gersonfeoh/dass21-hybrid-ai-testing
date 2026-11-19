# -*- coding: utf-8 -*-
"""LLM-RAG test proposer with a deterministic validator (Article 2, Section 6.4).

The LLM never supplies expected outputs: every proposed test is re-scored by the
reference oracle, and the LLM's own 'expected' block is only compared with it as a metric.

Retrieval is hybrid: BM25 and dense embeddings (Ollama /api/embed, e.g. bge-m3), fused by
reciprocal rank fusion and diversified by maximal marginal relevance (top-k = 5).
Generation uses the Ollama /api/chat endpoint with JSON output, a fixed temperature and seed.
"""
import json, os, re, sys, time, hashlib, urllib.request
import numpy as np
from rank_bm25 import BM25Okapi
import jsonschema
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s1 import ref_out

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = json.load(open(os.path.join(HERE, 'corpus.json'), encoding='utf-8'))

SCHEMA = {
    'type': 'object',
    'required': ['test_id', 'requirement_ids', 'intent', 'target_condition', 'items', 'expected', 'rationale'],
    'properties': {
        'test_id': {'type': 'string'},
        'requirement_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1},
        'intent': {'type': 'string'},
        'target_condition': {'type': 'object'},
        'items': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 3}, 'minItems': 21, 'maxItems': 21},
        'expected': {'type': 'object', 'required': ['c_D', 'c_A', 'c_S', 'G', 'F'],
                     'properties': {'c_D': {'type': 'integer'}, 'c_A': {'type': 'integer'}, 'c_S': {'type': 'integer'},
                                    'G': {'type': 'boolean'}, 'F': {'type': 'boolean'}}},
        'rationale': {'type': 'string'},
    },
}

PROMPT = """You are a software test designer for a DASS-21 scoring and triage engine.
Use ONLY the requirement excerpts below. Propose ONE new test case that targets the requirement {req}
and the condition: {cond}. Do not give any clinical advice or diagnosis.

Requirement excerpts:
{ctx}

Return ONLY one JSON object with the keys: test_id, requirement_ids (IDs of the excerpts you used),
intent, target_condition (booleans among N_D, N_A, N_S, C, I10, I17, I21, G, F), items (21 integers 0..3,
item 1 first), expected (c_D, c_A, c_S as category indices 0=Normal..4=Extremely Severe, G, F), rationale."""

TARGETS = [  # requirement id, condition text, condition dict checked by the validator
    ('REQ-GATE', 'gate open with all subscales Normal and no critical item at 3', dict(N_D=True, N_A=True, N_S=True, I10=False, I17=False, I21=False)),
    ('REQ-GATE', 'depression just above Normal while anxiety and stress stay Normal (gate closes)', dict(N_D=False, N_A=True, N_S=True)),
    ('REQ-GATE', 'anxiety just above Normal while depression and stress stay Normal', dict(N_D=True, N_A=False, N_S=True)),
    ('REQ-GATE', 'stress just above Normal while depression and anxiety stay Normal', dict(N_D=True, N_A=True, N_S=False)),
    ('REQ-FLAG', 'depression Severe or above with every critical item below 3', dict(C=True, I10=False, I17=False, I21=False)),
    ('REQ-FLAG', 'item 10 equal to 3 alone, depression below Severe', dict(C=False, I10=True, I17=False, I21=False)),
    ('REQ-FLAG', 'item 17 equal to 3 alone, depression below Severe', dict(C=False, I10=False, I17=True, I21=False)),
    ('REQ-FLAG', 'item 21 equal to 3 alone, depression below Severe', dict(C=False, I10=False, I17=False, I21=True)),
    ('REQ-GATE', 'gate open while the flag is raised by a critical item', dict(G=True, F=True)),
    ('REQ-SCORE', 'depression score at the largest attainable value below the Mild cut-point', dict(sD=8)),
    ('REQ-SCORE', 'anxiety score at the smallest attainable value at or above the Moderate cut-point', dict(sA=10)),
    ('REQ-SCORE', 'stress score at the largest attainable value below the Severe cut-point', dict(sS=24)),
]


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Retriever:
    """Hybrid BM25 + dense retrieval with reciprocal rank fusion and MMR."""

    def __init__(self, base_url=None, embed_model=None):
        self.docs = CORPUS
        self.tok = lambda t: re.findall(r'\w+', t.lower())
        self.bm25 = BM25Okapi([self.tok(d['text']) for d in self.docs])
        self.base_url, self.embed_model = base_url, embed_model
        self.E = self.embed([d['text'] for d in self.docs]) if embed_model else None

    def embed(self, texts):
        js = post(self.base_url.rstrip('/') + '/api/embed', {'model': self.embed_model, 'input': texts})
        E = np.array(js['embeddings'], float)
        return E / np.linalg.norm(E, axis=1, keepdims=True)

    def top(self, query, k=5, lam=0.7, rrf_k=60):
        bm = self.bm25.get_scores(self.tok(query))
        rank_bm = np.argsort(np.argsort(-bm))
        score = 1.0 / (rrf_k + rank_bm + 1)
        if self.E is not None:
            q = self.embed([query])[0]
            dense = self.E @ q
            rank_de = np.argsort(np.argsort(-dense))
            score = score + 1.0 / (rrf_k + rank_de + 1)
        sel, cand = [], list(np.argsort(-score))
        while cand and len(sel) < k:      # maximal marginal relevance on the fused score
            def mmr(i):
                if self.E is None or not sel:
                    return score[i]
                return lam * score[i] - (1 - lam) * max(float(self.E[i] @ self.E[j]) for j in sel) * score.max()
            best = max(cand, key=mmr)
            sel.append(best); cand.remove(best)
        return [self.docs[i] for i in sel]


def conditions(x):
    o = ref_out(np.array([x]))[0]
    sc = {s: 2 * sum(x[i - 1] for i in items) for s, items in (('D', (3, 5, 10, 13, 16, 17, 21)), ('A', (2, 4, 7, 9, 15, 19, 20)), ('S', (1, 6, 8, 11, 12, 14, 18)))}
    return dict(N_D=bool(o[0] == 0), N_A=bool(o[1] == 0), N_S=bool(o[2] == 0), C=bool(o[0] >= 3),
                I10=x[9] == 3, I17=x[16] == 3, I21=x[20] == 3, G=bool(o[3]), F=bool(o[4]),
                sD=sc['D'], sA=sc['A'], sS=sc['S']), o


def validate(raw_text, target, retrieved_ids, seen):
    """Returns (record, reason); reason is None for a valid test. A valid test passes the
    JSON schema and the item domain and is not a duplicate; everything else is measured."""
    m = re.search(r'\{.*\}', raw_text or '', re.S)
    if not m:
        return None, 'no_json'
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, 'invalid_json'
    try:
        jsonschema.validate(obj, SCHEMA)
    except jsonschema.ValidationError:
        return None, 'schema_or_domain'
    x = obj['items']
    if tuple(x) in seen:
        return None, 'duplicate'
    cond, o = conditions(x)
    exp = obj['expected']
    rec = dict(items=x, oracle=o.tolist(),
               oracle_agrees=[exp['c_D'], exp['c_A'], exp['c_S'], int(exp['G']), int(exp['F'])] == o.tolist(),
               target_met=all(cond[k] == v for k, v in target[2].items()),
               traceable=all(r in retrieved_ids for r in obj['requirement_ids']),
               cites_target=target[0] in obj['requirement_ids'])
    seen.add(tuple(x))
    return rec, None


def chat(base_url, model, prompt, temperature, seed, num_predict=512):
    t0 = time.time()
    js = post(base_url.rstrip('/') + '/api/chat', {
        'model': model, 'stream': False, 'format': 'json',
        'messages': [{'role': 'user', 'content': prompt}],
        'options': {'temperature': temperature, 'seed': seed, 'num_predict': num_predict, 'num_ctx': 4096}})
    return js['message']['content'], dict(prompt_tokens=js.get('prompt_eval_count', 0), output_tokens=js.get('eval_count', 0)), time.time() - t0


def build_prompt(retriever, t):
    docs = retriever.top(t[0] + ' ' + t[1], 5)
    ctx = '\n'.join('[%s] %s' % (d['id'], d['text']) for d in docs)
    return PROMPT.format(req=t[0], cond=t[1], ctx=ctx), [d['id'] for d in docs]

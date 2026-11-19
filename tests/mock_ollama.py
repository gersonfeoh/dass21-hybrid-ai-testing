# -*- coding: utf-8 -*-
"""Test double of the Ollama API (/api/tags, /api/embed, /api/chat) for checking the pipeline
code without a model. Its outputs are synthetic and are never research results."""
import json, re, random, hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np


def vec(t):
    v = np.zeros(64)
    for w in re.findall(r'\w+', t.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    return (v + 1e-3).tolist()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({'models': [{'name': 'mock:latest', 'digest': 'mock'}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/api/embed':
            return self._send({'embeddings': [vec(t) for t in body['input']]})
        o = body.get('options', {}); r = random.Random(o.get('seed', 0) * 7919 + len(body['messages'][0]['content']))
        k = r.random()
        if k < 0.1:
            txt = 'not json'
        elif k < 0.2:
            txt = json.dumps({'items': [5] * 21})
        else:
            x = [r.choice([0, 0, 0, 1, 2, 3]) for _ in range(21)]
            txt = json.dumps({'test_id': 't', 'requirement_ids': ['REQ-FLAG'], 'intent': 'i', 'target_condition': {},
                              'items': x, 'expected': {'c_D': 0, 'c_A': 0, 'c_S': 0, 'G': True, 'F': False}, 'rationale': 'r'})
        self._send({'message': {'content': txt}, 'prompt_eval_count': 600, 'eval_count': 150})


if __name__ == '__main__':
    import sys
    HTTPServer(('127.0.0.1', int(sys.argv[1]) if len(sys.argv) > 1 else 11435), H).serve_forever()

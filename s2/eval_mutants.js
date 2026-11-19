'use strict';
/*
 * Executes the original S2 module and every StrykerJS mutant on a batch of response vectors.
 * Usage: node eval_mutants.js <mutation.json> <vectors.bin> <out.bin>
 *   vectors.bin: int8, n x 21 ; out.bin: int8, (1 + M) x n x 5 (row 0 = original).
 * Each mutant is compiled in its own vm context; a mutant whose run exceeds the
 * timeout is recorded with output code -3 (detected by timeout, as in StrykerJS).
 */
const fs = require('fs'); const vm = require('vm'); const path = require('path');
const [rep, vecPath, outPath, mode] = process.argv.slice(2);  // mode 'kill' writes a JSON summary only
const report = JSON.parse(fs.readFileSync(rep, 'utf8'));
const fileKey = Object.keys(report.files)[0];
const original = report.files[fileKey].source;
const mutants = report.files[fileKey].mutants.slice().sort((a, b) => Number(a.id) - Number(b.id));
const lib = fs.readFileSync(path.join(__dirname, 'test', 'lib.js'), 'utf8');
const buf = fs.readFileSync(vecPath); const n = buf.length / 21;
const lineStarts = [0]; for (let i = 0; i < original.length; i++) if (original[i] === '\n') lineStarts.push(i + 1);
const off = (p) => lineStarts[p.line - 1] + p.column - 1;
function mutate(m) { const a = off(m.location.start), b = off(m.location.end); return original.slice(0, a) + m.replacement + original.slice(b); }
const runner = `
  var module = { exports: {} }; var exports = module.exports;
  (function (module, exports) { ${'__SRC__'} })(module, exports);
  var eng = module.exports;
  var lib = (function () { var module = { exports: {} }; ${lib.replace(/'use strict';/, '')} ; return module.exports; })();
  var out = new Int8Array(N * 5);
  for (var r = 0; r < N; r++) {
    var x = []; for (var j = 0; j < 21; j++) x.push(V[r * 21 + j]);
    var o = lib.keluaran(eng, x);
    for (var k = 0; k < 5; k++) out[r * 5 + k] = o[k];
  }
  out;`;
function run(src) {
  const ctx = vm.createContext({ V: new Int8Array(buf.buffer, buf.byteOffset, buf.length), N: n, Int8Array, Number, isNaN, JSON, Object });
  try {
    return vm.runInContext(runner.replace('__SRC__', () => src), ctx, { timeout: Math.floor(Math.max(5000, n / 2)) });
  } catch (e) {
    if (process.env.DEBUG) console.error(String(e).slice(0, 300));
    const t = new Int8Array(n * 5); t.fill(String(e && e.code) === 'ERR_SCRIPT_EXECUTION_TIMEOUT' ? -3 : -4); return t;
  }
}
const sources = [original].concat(mutants.map(mutate));
if (mode === 'orig') {
  const r = run(original); fs.writeFileSync(outPath, Buffer.from(r.buffer, r.byteOffset, r.length));
} else if (mode === 'kill') {
  const o = run(original); if (o[0] < -2) throw new Error('original failed on the batch'); const summ = [];
  mutants.forEach((m, i) => {
    const r = run(sources[i + 1]); let k = 0, first = -1;
    for (let v = 0; v < n; v++) { let d = false; for (let j = 0; j < 5; j++) if (r[v * 5 + j] !== o[v * 5 + j]) { d = true; break; } if (d) { k++; if (first < 0) first = v; } }
    summ.push({ id: m.id, kills: k, first: first });
  });
  fs.writeFileSync(outPath, JSON.stringify(summ));
} else {
  const out = Buffer.alloc((mutants.length + 1) * n * 5);
  sources.forEach((s, i) => { const r = run(s); Buffer.from(r.buffer, r.byteOffset, r.length).copy(out, i * n * 5); });
  fs.writeFileSync(outPath, out);
}
fs.writeFileSync(outPath + '.ids.json', JSON.stringify(mutants.map(m => ({ id: m.id, mutator: m.mutatorName, replacement: m.replacement,
  line: m.location.start.line, col: m.location.start.column, endLine: m.location.end.line, endCol: m.location.end.column, stryker_status: m.status,
  original: original.slice(off(m.location.start), off(m.location.end)) }))));
console.log('mutants', mutants.length, 'vectors', n);

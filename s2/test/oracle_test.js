'use strict';
// Runs the 35-vector oracle of Article 1 (baseline B0) against src/dass21.js.
const fs = require('fs');
const path = require('path');
const eng = require('../src/dass21.js');
const { keluaran } = require('./lib.js');
const EN = { 'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely Severe': 4 };
const rows = fs.readFileSync(path.join(__dirname, 'oracle.csv'), 'utf8').trim().split(/\r?\n/);
const h = rows[0].split(',');
let gagal = 0;
for (const line of rows.slice(1)) {
  const c = line.split(','); const r = {}; h.forEach((k, i) => { r[k] = c[i]; });
  const x = []; for (let i = 1; i <= 21; i++) x.push(Number(r['b' + String(i).padStart(2, '0')]));
  const harap = [EN[r.harapan_D], EN[r.harapan_A], EN[r.harapan_S], r.harapan_gerbang === 'terbuka' ? 1 : 0, r.harapan_penanda === 'ya' ? 1 : 0];
  const dapat = keluaran(eng, x);
  if (JSON.stringify(dapat) !== JSON.stringify(harap)) { gagal++; if (!process.env.QUIET) console.log('FAIL', r.kasus, dapat, harap); }
}
if (!process.env.QUIET) console.log('cases', rows.length - 1, 'failed', gagal);
process.exit(gagal ? 1 : 0);

'use strict';
// Maps the platform outputs to the five oracle outputs: categories D, A, S (0..4), gate (0/1), flag (0/1).
const LABEL = { 'Normal': 0, 'Ringan': 1, 'Sedang': 2, 'Berat': 3, 'Sangat Berat': 4 };
function lab(x) { return Object.prototype.hasOwnProperty.call(LABEL, x) ? LABEL[x] : 9; }
function keluaran(eng, x) {
  try {
    const r = eng.nilaiDass(x.slice());
    if (!r || !r.ok || !r.kat) return [-1, -1, -1, -1, -1];
    const row = { kat_D: r.kat.D, kat_A: r.kat.A, kat_S: r.kat.S };
    const g = eng.bolehChatDari([row]);
    return [lab(r.kat.D), lab(r.kat.A), lab(r.kat.S), (g && g.boleh === true) ? 1 : 0, r.flag ? 1 : 0];
  } catch (e) {
    return [-2, -2, -2, -2, -2];
  }
}
module.exports = { keluaran, LABEL };

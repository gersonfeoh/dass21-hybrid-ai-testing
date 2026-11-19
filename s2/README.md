# Subject S2 (code-level mutants)

S2 is the scoring, triage-flag and chat-eligibility logic of the DASS-21 platform (Google Apps Script),
copied verbatim into a JavaScript module with all I/O removed, and mutated with StrykerJS 10.1.1.

The module (`src/dass21.js`) and the StrykerJS report (`reports/mutation.json`, which embeds the module
source) are **not distributed**, because they contain production source code. SHA-256 of the module used
in the experiments: `4d9b4f6ca78903d2fe0fe40c72e42a5c7cf4716bb39382366991d5f99f58c5d4`.

What is distributed: the evaluator (`eval_mutants.js`), the output mapping (`test/lib.js`), the 35-vector
oracle test used as the StrykerJS test command, the per-mutant review (`../data/s2_labels.json`: mutator,
line, family, equivalence review), and every S2 result in `../data/`.

Authors with access to the private bundle (`s2_private.zip`) can unpack it into this folder; all scripts
then include S2 automatically (`evaluate_lib.s2_available()`).

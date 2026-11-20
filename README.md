# dass21-hybrid-ai-testing

Experiment code, data, and logs for the article *SMT-Guided, Lattice-Aware Search-Based Testing of a DASS-21
Screening and Triage Engine, with an Oracle-Subordinate LLM-RAG Baseline*, the second article of the DASS-21 research
line at Universitas Dhyana Pura (with Vellore Institute of Technology). The approach combines (i) SMT verification of
safety invariants with Z3 and (ii) a multi-objective genetic algorithm (NSGA-II) with a lattice-aware genotype under
hard safety constraints derived by the solver. (iii) Fault-family diagnosis with Random Forest and XGBoost is a
supporting analysis, and (iv) an LLM-RAG test proposer whose every output is validated by a deterministic oracle is
evaluated as a baseline.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gersonfeoh/dass21-hybrid-ai-testing/blob/main/colab/DASS21_LLM_RAG_Colab.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24385583.svg)](https://doi.org/10.5281/zenodo.24385583)

Status (20 November 2026): every component has been run. The LLM-RAG experiment (B4, B5, A1, A3) was run with the
Colab notebook on a T4 GPU (Ollama 0.40.16, qwen2.5:7b-instruct Q4_K_M, bge-m3); its logs are in `results_llm/`.
Release v1.0.0, used for every result of the article, is archived on Zenodo: https://doi.org/10.5281/zenodo.24385584
(all versions: https://doi.org/10.5281/zenodo.24385583). The first article (lattice-aware boundary and mutation testing, JUTI) and its reference engine are
archived at https://doi.org/10.5281/zenodo.24131491; this repository reuses that engine in `article1/`.

## Main results (S1 = specification-level mutants, S2 = StrykerJS mutants of the platform code)

| Configuration | Held-out S1 MS | S2 MS | Size after minimisation | Seeds with all 12 safety obligations |
|---|---|---|---|---|
| B0 35-vector oracle (Article 1) | 100.0% | 99.4% | 27 | 1/1 |
| B1 random search | 98.0% | 90.4% | 14 | 0/30 |
| B2 raw-genotype GA | 100.0% | 99.4% | 15 | 0/30 |
| B3 lattice-aware GA + Z3 | 100.0% | 99.4% | 17 | 30/30 |
| A1' B3 without Z3 | 100.0% | 99.4% | 16 | 0/30 |
| A3' B3 with raw genotype | 98.0% | 97.8% | 14 | 0/30 |
| B4 LLM-RAG tests only | 22.0% | 50.0% | 2 | 0/30 |
| B5 B3 + LLM-RAG tests | 100.0% | 99.4% | 18 | 30/30 |
| A1 B5 without Z3 | 100.0% | 99.4% | 16 | 0/30 |
| A3 B5 with raw genotype | 98.0% | 98.4% | 14 | 0/30 |

Medians over 30 seeds. Z3 proved every invariant except gate safety: 6.50% of gate-opening response patterns
also raise the triage flag. Fault-family diagnosis (StratifiedGroupKFold by fault site): XGBoost macro-F1 0.740,
hand-written rules 0.582; transfer to S2: Random Forest 0.811.

LLM-RAG (372 calls): only 25.0% (temperature 0) and 14.2% (temperature 0.7) of responses passed the deterministic
validator, mostly because of a wrong number of items or values outside 0 to 3 (a post hoc lenient validator gives
33.3% and 20.8%). At temperature 0.7 the scores claimed by the model agreed with the oracle for 5.9% of valid tests,
and no valid test killed a mutant missed by B0. B5 does not differ from B3 (A2) in any metric, so the LLM tests add
no killed mutants on this subject. `results_llm/colab_original/` holds the Colab output before the offline S2
re-measurement; the S1 metrics are identical.

## Repository layout

| Path | Content |
|---|---|
| `s1.py` | Parametric reference engine, specification-level mutant model, vectorised evaluator, Z3 engine model |
| `build_pool.py` | S1 mutant pool (165 first-order, 500 second-order), exact equivalence with Z3, 70:30 split by fault site |
| `verify.py` | Safety invariants P1 to P8 and candidate G' with Z3; derivation of the 12 safety test obligations |
| `ga.py`, `pilot.py`, `main_runs.py` | NSGA-II suite generation (raw and lattice-aware genotypes), pilot, 150 main runs |
| `evaluate_lib.py`, `evaluate_runs.py`, `stats.py`, `extra.py` | Measurement on held-out pools, greedy minimisation, Mann-Whitney U, Vargha-Delaney A12, Holm |
| `diagnose.py` | Fault-family diagnosis: majority, rules, logistic regression, Random Forest, XGBoost |
| `equiv_s2.py`, `label_s2.py`, `s2/` | S2 mutant screening, equivalence review, StrykerJS configuration and evaluator |
| `llm_rag/` | Hybrid BM25 + dense retrieval, prompt, JSON schema, deterministic validator, LLM experiment, post hoc lenient validator (`sensitivity.py`), per-mutant detail (`extra_llm.py`) |
| `results_llm/` | LLM-RAG log (prompts, responses, tokens, latency), validated tests, Table 11 metrics, 90 GA runs, evaluation, statistics, Colab environment |
| `colab/DASS21_LLM_RAG_Colab.ipynb` | Google Colab notebook that installs Ollama and runs the LLM-RAG experiment |
| `data/`, `runs/`, `pilot/` | Mutant manifests, Z3 results, run logs, measurements, statistics, diagnosis results |
| `tests/mock_ollama.py` | Test double of the Ollama API used only to check the code paths |

## Reproduce

```
pip install -r requirements.txt
python3 build_pool.py && python3 verify.py
python3 pilot.py && python3 main_runs.py 2
python3 evaluate_runs.py && python3 stats.py && python3 extra.py && python3 diagnose.py
```

The LLM-RAG experiment: open the Colab notebook (T4 GPU) or, with a local Ollama server,
`python3 llm_rag/llm_experiment.py --model qwen2.5:7b-instruct --embed-model bge-m3 --seeds 30`.
The script resumes from `results_llm/llm_log.jsonl`, so with the published log it re-measures every metric without an
LLM call. `python3 llm_rag/sensitivity.py` reproduces the lenient-validator analysis. S2 measurement and
`extra.py`/`extra_llm.py` need the private S2 module.

## Confidentiality and scope

The S2 module (`s2/src/dass21.js`) and its StrykerJS report contain production source code of the platform and are
not distributed; every S2 result is in `data/`. All inputs are synthetic response vectors; no student data are used.
The results are software testing results and are not a clinical validation of DASS-21 or of the triage rules.
Candidate rule changes such as G' = G and not F require a clinical decision.

## License and citation

MIT License. See `CITATION.cff`.

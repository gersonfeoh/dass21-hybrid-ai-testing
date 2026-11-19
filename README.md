# Hybrid AI-Driven Testing and Verification Framework for DASS-21 Screening and Triage Engines

Experiment code, data, and logs for the second article of the DASS-21 research line at Universitas Dhyana Pura
(with Vellore Institute of Technology). The framework combines (i) SMT verification of safety invariants with Z3,
(ii) a multi-objective genetic algorithm (NSGA-II) with a lattice-aware genotype under hard safety constraints,
(iii) fault-family diagnosis with Random Forest and XGBoost, and (iv) an LLM-RAG test proposer whose every output
is validated by a deterministic oracle.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gersonfeoh/dass21-hybrid-ai-testing/blob/main/colab/DASS21_LLM_RAG_Colab.ipynb)

Status (20 November 2026): components (i) to (iii) have been run; the LLM-RAG experiment (B4, B5, A1, A3) runs in the
Colab notebook. The first article (lattice-aware boundary and mutation testing, JUTI) and its reference engine are
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

Medians over 30 seeds. Z3 proved every invariant except gate safety: 6.50% of gate-opening response patterns
also raise the triage flag. Fault-family diagnosis (StratifiedGroupKFold by fault site): XGBoost macro-F1 0.740,
hand-written rules 0.582; transfer to S2: Random Forest 0.811.

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
| `llm_rag/` | Hybrid BM25 + dense retrieval, prompt, JSON schema, deterministic validator, LLM experiment |
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

## Confidentiality and scope

The S2 module (`s2/src/dass21.js`) and its StrykerJS report contain production source code of the platform and are
not distributed; every S2 result is in `data/`. All inputs are synthetic response vectors; no student data are used.
The results are software testing results and are not a clinical validation of DASS-21 or of the triage rules.
Candidate rule changes such as G' = G and not F require a clinical decision.

## License and citation

MIT License. See `CITATION.cff`.

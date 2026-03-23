# Contributing to Geometric Latent Biopsy

Thank you for your interest in contributing. This document outlines how to get involved, what kinds of contributions are most useful, and the conventions the project follows.

## Ways to Contribute

**Bug reports and reproducibility issues.** If you run the pipeline on a different model or hardware configuration and get unexpected results, please open an issue with the model name, hardware, Python/PyTorch versions, and the full traceback or unexpected output.

**New model evaluations.** Running `run_model.py` on models beyond Qwen2.5-0.5B-Instruct and sharing the results (or a PR adding them to the repo) is one of the highest-value contributions right now. Cross-model generalisation is the main open question.

**Methodological extensions.** Ideas that have been discussed but not yet implemented include multi-turn analysis (beyond single-prompt, last-token), alternative manifold modelling (e.g., replacing the GMM with a normalizing flow), layer-aggregation strategies beyond concatenation, and adversarial robustness testing (prompts engineered to stay on-manifold).

**Documentation and clarity.** Improvements to docstrings, explanations of the geometry, or tutorial notebooks are always welcome.

## Development Setup

```bash
git clone https://github.com/isaac-6/geometric-latent-biopsy.git
cd geometric-latent-biopsy
pip install torch transformers scikit-learn matplotlib numpy pandas scipy datasets requests pytest
```

Verify everything works:

```bash
pytest test_theta.py test_extraction.py -v
```

## Submitting Changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Run the existing tests and confirm they pass. If you're adding new functionality, add corresponding tests.
4. Open a pull request against `main`. In the PR description, explain what the change does and why.

## Code Conventions

**Style.** The codebase uses standard Python conventions: type hints where practical, docstrings on public functions and classes, and descriptive variable names. There is no strict formatter enforced, but try to match the existing style.

**Architecture.** The two core modules are `extraction.py` (model interaction) and `theta.py` (geometry and scoring). Everything else is a script that composes these. If you're adding a new analysis, it should import from these core modules rather than duplicating their logic.

**Tests.** Tests live in `test_extraction.py` and `test_theta.py`. They cover the mathematical core (angle computation edge cases, zero-vector handling, high-dimensional precision) and the extraction pipeline. New features should include tests that verify correctness without requiring a GPU or large model download — use synthetic tensors where possible.

**Reproducibility.** All scripts accept a `--seed` argument. Random operations should use this seed. The `run_model.py` pipeline writes a `manifest.json` recording exact parameters — any new pipeline step should update this manifest.

## Evaluation Integrity

The project is careful about train/eval contamination. If you modify the evaluation pipeline, please preserve these invariants:

- Normative fit-set prompts are never scored by the biomarker they trained.
- Benign-aggressive prompts are never used for fitting under either strategy.
- The `DataSplit` object in `evaluate_biomarker.py` is the single source of truth for all splits.

## Reporting Issues

When opening an issue, please include:

- What you were trying to do
- The command you ran
- The full error output or unexpected behaviour
- Your environment (Python version, PyTorch version, GPU/CPU, OS)

## Code of Conduct

Be respectful and constructive. This is a research project exploring AI safety methods. Good-faith disagreement about methodology is welcome; hostility is not.
# Geometric Latent Biopsy

A zero-shot geometric anomaly detector for LLM residual streams. This method identifies harmful or anomalous prompts by measuring angular deviations from a normative (safe) manifold in the model's hidden representations, without ever seeing harmful examples at training time.

## Core Idea

Safety-aligned LLMs develop structured internal representations where safe prompts cluster on a sub-manifold of the residual stream. Harmful prompts deviate from this manifold in geometrically detectable ways.

The **Theta Biomarker** captures this by:

1. Extracting last-token hidden states across all layers for a set of normative (safe) prompts.
2. Computing principal directions (PC1…PCK) of the normative distribution via PCA.
3. For each new prompt, measuring the angular deviation (θ) to each principal direction, producing a K-dimensional angle vector.
4. Modelling the normative angle distribution with a Gaussian Mixture Model (GMM) fitted on circular-embedded (sin θ, cos θ) features.
5. Scoring new prompts by their negative log-likelihood under this GMM. Higher scores indicate greater deviation from the safe manifold.

When K=1 and the GMM has one component, this reduces to a single-angle, single-centroid method. The framework generalises to multiple directions and mixture components for richer manifold modelling.

### Two Reference Strategies

| Strategy | Fit data | Score interpretation | Use case |
|---|---|---|---|
| **`normative_ref`** (zero-shot) | Safe prompts only | Higher = farther from safe manifold | No harmful data needed at fit time |
| **`harmful_ref`** (supervised, experimental) | Harmful prompts | Higher = closer to harmful manifold | When harmful examples are available |

## Repository Structure

```
geometric-latent-biopsy/
├── extraction.py              # LatentExtractor — hidden state extraction from HF models
├── theta.py                   # ThetaBiomarker — core geometric anomaly detector
├── download_datasets.py       # Fetches Alpaca-Cleaned, AdvBench, XSTest
├── run_model.py               # Full pipeline: download → auto-tune → evaluate → plot
├── evaluate_biomarker.py      # Systematic evaluation (AUROC, PR curves, statistics)
├── stability_analysis.py      # Normative set size sensitivity analysis
├── analyze_topology.py        # Pairwise angular distance topology analysis
├── plot_pc1_reference.py      # PC1 reference projection visualisation
├── plot_theta_phi_full.py     # Theta-phi plane with full evaluation datasets
├── plot_theta_phi_plane.py    # Theta-phi orthogonal projection (small demo)
├── run_first_biopsy.py        # Minimal "hello world" biopsy example
├── test_extraction.py         # Tests for activation extraction
└── test_theta.py              # Tests for theta computation and biomarker fitting
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- A HuggingFace-compatible causal language model

### Setup

```bash
git clone https://github.com/isaac-6/geometric-latent-biopsy.git
cd geometric-latent-biopsy

pip install torch transformers scikit-learn matplotlib numpy pandas scipy datasets requests
```

For GPU acceleration (recommended for larger models):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Minimal Example

```bash
python run_first_biopsy.py
```

This fits a biomarker on five safe prompts and compares theta profiles for a safe vs. harmful prompt across all layers. Output: `results/figures/first_biopsy_theta.png`.

### Full Pipeline (Single Command)

```bash
python run_model.py --model Qwen/Qwen2.5-0.5B-Instruct --strategy normative_ref
```

This will:
1. Download datasets (Alpaca-Cleaned, AdvBench, XSTest) if not present
2. Auto-tune the normative fit-N via plateau analysis
3. Run the full evaluation with per-layer AUROC, dimension ablation, PR curves, and statistical tests
4. Generate theta-phi projection plots

All outputs land in `results/Qwen__Qwen2.5-0.5B-Instruct/` with a reproducibility manifest.

### Step-by-Step

```bash
# 1. Download datasets
python download_datasets.py --normative-n 500 --seed 42

# 2. Stability analysis (how many normative prompts are enough?)
python stability_analysis.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --layers 0 6 12 19 22

# 3. Full evaluation
python evaluate_biomarker.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --normative-fit-n 200 \
    --strategy normative_ref

# 4. Theta-phi visualisation
python plot_theta_phi_full.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --layer 19 \
    --strategy normative_ref
```

## Datasets

| Dataset | Role | Source |
|---|---|---|
| **Alpaca-Cleaned** | Normative (safe) prompts | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) |
| **AdvBench** | Harmful prompts | [Zou et al., 2023](https://github.com/llm-attacks/llm-attacks) |
| **XSTest** | Benign-aggressive (safe but edgy) | [Röttger et al., 2023](https://github.com/paul-rottger/xstest) |

Benign-aggressive prompts (e.g., "How do I kill a running process in Linux?") are never used for fitting and serve as the hard-negative evaluation set.

## Key Configuration

| Parameter | Default | Description |
|---|---|---|
| `--n-directions` | 1 | Number of PCA directions (K). K=2 often improves discrimination. |
| `--top-d-dims` | None | Restrict to top-D dimensions by normative variance before PCA. |
| `--normative-fit-n` | 200 | Number of safe prompts for fitting. Stability analysis shows AUROC plateaus around N≈200. |
| `--strategy` | `normative_ref` | Reference strategy: `normative_ref`, `harmful_ref`, or `both`. |
| `--auroc-plateau-tol` | 0.01 | Tolerance for auto-tuning fit-N (1 percentage point). |

## Evaluation Outputs

Under `results/eval/<strategy>/`:

- `auroc_by_layer.png` — Per-layer AUROC across K directions with cosine and L2 baselines
- `auroc_ablation_dim.png` — Dimension pruning ablation (normative_ref only)
- `score_distributions.png` — Violin plots of anomaly scores by category
- `precision_recall.png` — PR curves with 90%/95% recall operating points
- `stats_summary.csv` — AUROC, AUPRC, precision at recall targets, Mann-Whitney U, rank-biserial r

## Using the Biomarker in Your Own Code

```python
import torch
from extraction import LatentExtractor
from theta import ThetaBiomarker

# Extract activations
extractor = LatentExtractor("Qwen/Qwen2.5-0.5B-Instruct")
safe_acts = torch.stack([
    extractor.get_last_token_activations(p)
    for p in safe_prompts
])  # (N, layers, hidden_dim)

# Fit on safe prompts only
biomarker = ThetaBiomarker(n_directions=2, layer_indices=[19])
biomarker.fit(safe_acts)

# Score a new prompt
new_act = extractor.get_last_token_activations("some prompt")
score = biomarker.score(new_act)  # higher = more anomalous
```

## Tests

```bash
pytest test_theta.py test_extraction.py -v
```

Tests cover core angle math (identical, opposite, orthogonal, zero vectors), zero-centroid rejection, and high-dimensional float32 precision bounds.

## Limitations

- Evaluated primarily on Qwen2.5-0.5B and Qwen3.5-0.8B (both base and instruct); cross-model generalisation is an open question.
- The method detects geometric deviation, not semantic harm. Adversarial prompts engineered to stay on-manifold may evade detection.
- Benign-aggressive prompts can produce elevated scores under normative_ref, reflecting surface-level similarity to harmful language rather than genuine risk.
- Single-prompt, last-token analysis; multi-turn and mid-sequence dynamics are not captured.

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this work in your research, please cite it — see [CITATION.cff](CITATION.cff) for details.
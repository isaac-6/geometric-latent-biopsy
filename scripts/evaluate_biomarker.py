"""
evaluate_biomarker.py
---------------------
Systematic evaluation of the ThetaBiomarker across:

    * All layers independently
    * Ablation over number of reference directions (K = 1..4)
    * Ablation over dimension pruning (top-D by normative variance)
    * Baselines: cosine similarity to centroid, L2 norm, random direction
    * Statistical tests: Mann-Whitney U + effect size (rank-biserial r)
    * Classification: AUROC, AUPRC, and threshold-free metrics

Outputs (under results/eval/):
    auroc_by_layer.png            per-layer AUROC for each K
    auroc_ablation_K.png          best-layer AUROC vs K (direction count)
    auroc_ablation_dim.png        best-layer AUROC vs top-D dimensions kept
    score_distributions.png       violin plots at best layer
    stats_summary.csv             Mann-Whitney U, effect size, AUROC table

Usage
-----
    python scripts/evaluate_biomarker.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --normative-file data/raw/normative.txt \
        --harmful-file data/raw/harmful.txt \
        --benign-agg-file data/raw/benign_aggressive.txt \
        --normative-n 200 \
        --harmful-n 200 \
        --benign-agg-n 200 \
        --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import PolyCollection
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

# -- local imports -----------------------------------------------------------
# Resolve src/ relative to this script regardless of cwd, so the runtime
# interpreter finds the modules.  Pylance is pointed here via pyrightconfig.json
# (extraPaths: ["src"]) in the repo root — see note below.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction import LatentExtractor           # type: ignore[import-untyped]
from theta import ThetaBiomarker, compute_theta_core  # type: ignore[import-untyped]

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path("results/eval")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prompts(path: str, n: int, seed: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(seed)
    n = min(n, len(lines))
    return random.sample(lines, n)


# ---------------------------------------------------------------------------
# Baseline scorers
# ---------------------------------------------------------------------------

def score_cosine_to_centroid(
    acts: torch.Tensor,          # (N, L, D)
    normative_acts: torch.Tensor,# (M, L, D)
    layer: int,
) -> np.ndarray:
    """1 - cosine similarity to normative centroid (higher = more anomalous)."""
    centroid = normative_acts[:, layer, :].mean(dim=0, keepdim=True)  # (1, D)
    X = acts[:, layer, :]  # (N, D)
    cos = torch.nn.functional.cosine_similarity(X, centroid.expand_as(X), dim=-1)
    return (1 - cos).cpu().numpy()


def score_l2_norm(acts: torch.Tensor, layer: int) -> np.ndarray:
    """L2 norm of the activation (deviation from origin)."""
    return torch.linalg.norm(acts[:, layer, :], dim=-1).cpu().numpy()


def score_random_direction(
    acts: torch.Tensor, layer: int, seed: int
) -> np.ndarray:
    """Theta against a random unit vector — null hypothesis baseline."""
    torch.manual_seed(seed)
    D = acts.shape[-1]
    rand_vec = torch.randn(1, D, dtype=acts.dtype, device=acts.device)
    rand_vec = rand_vec / torch.linalg.norm(rand_vec, dim=-1, keepdim=True)
    X = acts[:, layer, :]
    thetas = compute_theta_core(X, rand_vec.expand(X.shape[0], -1))
    return thetas.cpu().numpy()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size for Mann-Whitney U: rank-biserial correlation ∈ [-1, 1]."""
    n1, n2 = len(x), len(y)
    U, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(1 - (2 * U) / (n1 * n2))


def compute_auroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """
    AUROC where `scores_pos` (harmful) should have higher scores than
    `scores_neg` (safe/normative).
    """
    y_true  = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    y_score = np.concatenate([scores_neg, scores_pos])
    if np.isnan(y_score).any():
        return float("nan")
    # roc_auc_score returns np.floating; cast to plain float for type safety.
    return float(roc_auc_score(y_true, y_score))


def compute_auprc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    y_true  = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    y_score = np.concatenate([scores_neg, scores_pos])
    if np.isnan(y_score).any():
        return float("nan")
    return float(average_precision_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ---- Load prompts ----
    print("Loading prompts...")
    normative_prompts = load_prompts(args.normative_file, args.normative_n, args.seed)
    harmful_prompts   = load_prompts(args.harmful_file,   args.harmful_n,   args.seed)
    benign_prompts    = load_prompts(args.benign_agg_file, args.benign_agg_n, args.seed)

    print(f"  Normative : {len(normative_prompts)}")
    print(f"  Harmful   : {len(harmful_prompts)}")
    print(f"  Benign-Agg: {len(benign_prompts)}")

    # ---- Extract activations ----
    print(f"\nLoading model: {args.model}")
    extractor = LatentExtractor(args.model)
    L = extractor.num_layers

    def extract_batch(prompts):
        return torch.stack([extractor.get_last_token_activations(p) for p in prompts])

    print("Extracting normative activations...")
    norm_acts    = extract_batch(normative_prompts)   # (M, L, D)
    print("Extracting harmful activations...")
    harmful_acts = extract_batch(harmful_prompts)     # (N, L, D)
    print("Extracting benign-aggressive activations...")
    benign_acts  = extract_batch(benign_prompts)      # (N, L, D)

    layers = list(range(L))

    # ==================================================================
    # Experiment 1: Per-layer AUROC — ablation over K directions
    # ==================================================================
    print("\n[Exp 1] Per-layer AUROC across K directions...")

    K_values = [1, 2, 3, 4]
    auroc_harmful_by_K   = {K: [] for K in K_values}
    auroc_benign_by_K    = {K: [] for K in K_values}

    for K in K_values:
        print(f"  K={K} ...", end="", flush=True)
        for layer in layers:
            bm = ThetaBiomarker(
                n_directions=K,
                n_gmm_components=1,
                layer_indices=[layer],
            )
            bm.fit(norm_acts)

            norm_scores    = bm.score_batch(norm_acts)
            harmful_scores = bm.score_batch(harmful_acts)
            benign_scores  = bm.score_batch(benign_acts)

            auroc_harmful_by_K[K].append(compute_auroc(norm_scores, harmful_scores))
            auroc_benign_by_K[K].append(compute_auroc(norm_scores, benign_scores))

        print(" done")

    # ---- Baselines (computed once) ----
    print("  Computing baselines...")
    auroc_cos   = []
    auroc_l2    = []
    auroc_rand  = []

    for layer in layers:
        norm_cos   = score_cosine_to_centroid(norm_acts,    norm_acts, layer)
        harm_cos   = score_cosine_to_centroid(harmful_acts, norm_acts, layer)
        auroc_cos.append(compute_auroc(norm_cos, harm_cos))

        norm_l2  = score_l2_norm(norm_acts,    layer)
        harm_l2  = score_l2_norm(harmful_acts, layer)
        auroc_l2.append(compute_auroc(norm_l2, harm_l2))

        norm_rd  = score_random_direction(norm_acts,    layer, args.seed)
        harm_rd  = score_random_direction(harmful_acts, layer, args.seed)
        auroc_rand.append(compute_auroc(norm_rd, harm_rd))

    # ---- Plot ----
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, auroc_by_K, title_suffix in zip(
        axes,
        [auroc_harmful_by_K, auroc_benign_by_K],
        ["Harmful (AdvBench)", "Benign-Aggressive (XSTest)"],
    ):
        for K, c in zip(K_values, colors):
            ax.plot(layers, auroc_by_K[K], label=f"K={K} directions", color=c, lw=2)

        ax.plot(layers, auroc_cos,  "--", color="purple", alpha=0.7, label="Cosine (baseline)")
        ax.plot(layers, auroc_l2,   "--", color="brown",  alpha=0.7, label="L2-norm (baseline)")
        ax.axhline(0.5, color="gray", linestyle=":", label="Random baseline (0.5)")

        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC (normative vs. target)")
        ax.set_title(f"Per-Layer AUROC — {title_suffix}")
        ax.legend(fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "auroc_by_layer.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()

    # ==================================================================
    # Experiment 2: Dimension pruning ablation at best layer
    # ==================================================================
    print("\n[Exp 2] Dimension pruning ablation...")

    D = norm_acts.shape[-1]
    # Find best layer for K=1 (harmful AUROC)
    best_layer_K1 = int(np.argmax(auroc_harmful_by_K[1]))
    print(f"  Best layer (K=1, harmful AUROC): {best_layer_K1}")

    dim_fractions: list[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    # Build the final list in one shot so the inferred type is list[int | None]
    # from the start.  Assigning sorted(...) first yields list[int], and list is
    # invariant, so .append(None) would be a type error even with an explicit
    # annotation (Pylance enforces invariance at the assignment site).
    _dim_ints: list[int] = sorted(set(max(1, int(f * D)) for f in dim_fractions))
    dim_values: list[int | None] = [*_dim_ints, None]  # None = all dimensions

    auroc_dim_harmful  = []
    auroc_dim_benign   = []
    dim_labels         = []

    for top_d in dim_values:
        bm = ThetaBiomarker(
            n_directions=2,
            n_gmm_components=1,
            top_d_dims=top_d,
            layer_indices=[best_layer_K1],
        )
        bm.fit(norm_acts)
        ns  = bm.score_batch(norm_acts)
        hs  = bm.score_batch(harmful_acts)
        bs  = bm.score_batch(benign_acts)
        auroc_dim_harmful.append(compute_auroc(ns, hs))
        auroc_dim_benign.append(compute_auroc(ns, bs))
        dim_labels.append(str(top_d) if top_d is not None else "all")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(dim_labels))
    ax.bar(x - 0.2, auroc_dim_harmful, 0.35, label="vs Harmful (AdvBench)", color="#d62728", alpha=0.8)
    ax.bar(x + 0.2, auroc_dim_benign,  0.35, label="vs Benign-Agg (XSTest)", color="#2ca02c", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle=":", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"top-{l}" if l != "all" else "all dims" for l in dim_labels])
    ax.set_xlabel("Dimensions retained (by normative variance)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Dimension Pruning Ablation — Layer {best_layer_K1}, K=2")
    ax.legend()
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = RESULTS_DIR / "auroc_ablation_dim.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()

    # ==================================================================
    # Experiment 3: Score distributions + statistical tests at best layer
    # ==================================================================
    print("\n[Exp 3] Score distributions and statistical tests...")

    # Use best config: K that gave highest harmful AUROC at its best layer
    best_K        = max(K_values, key=lambda K: max(auroc_harmful_by_K[K]))
    best_layer    = int(np.argmax(auroc_harmful_by_K[best_K]))
    print(f"  Best config: K={best_K}, layer={best_layer}")

    bm_best = ThetaBiomarker(
        n_directions=best_K,
        n_gmm_components=1,
        layer_indices=[best_layer],
    )
    bm_best.fit(norm_acts)

    scores_norm    = bm_best.score_batch(norm_acts)
    scores_harmful = bm_best.score_batch(harmful_acts)
    scores_benign  = bm_best.score_batch(benign_acts)

    # ---- Violin plot — all four distributions ----
    fig, ax = plt.subplots(figsize=(10, 6))

    # "rest" = normative + benign-agg combined (the real-world negative class)
    scores_rest = np.concatenate([scores_norm, scores_benign])

    data   = [scores_norm, scores_harmful, scores_benign, scores_rest]
    labels = [
        "Normative\n(Alpaca)",
        "Harmful\n(AdvBench)",
        "Benign-Agg\n(XSTest)",
        "Rest\n(norm+benign)",
    ]
    colors_v = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    parts = ax.violinplot(data, positions=list(range(len(data))), showmedians=True)
    bodies = cast(list[PolyCollection], parts["bodies"])
    for pc, c in zip(bodies, colors_v):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Anomaly Score (−log p)")
    ax.set_title(
        f"Score Distributions — Layer {best_layer}, K={best_K}\n"
        f"AUROC harm/norm: {compute_auroc(scores_norm, scores_harmful):.3f}  |  "
        f"harm/benign: {compute_auroc(scores_benign, scores_harmful):.3f}  |  "
        f"harm/rest: {compute_auroc(scores_rest, scores_harmful):.3f}"
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = RESULTS_DIR / "score_distributions.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()

    # ---- Statistical tests ----
    # All comparisons: the second element is (negative_class, positive_class)
    comparisons: list[tuple[str, np.ndarray, np.ndarray]] = [
        # Classic: normative is the negative class
        ("normative_vs_harmful",    scores_norm,   scores_harmful),
        ("normative_vs_benign_agg", scores_norm,   scores_benign),
        # Operational: can we distinguish harmful from surface-similar safe?
        ("harmful_vs_benign_agg",   scores_benign, scores_harmful),
        # Real-world: harmful vs everything that should NOT be flagged
        ("harmful_vs_rest",         scores_rest,   scores_harmful),
    ]

    rows = []
    for label, scores_neg, scores_pos in comparisons:
        U, p = mannwhitneyu(scores_neg, scores_pos, alternative="two-sided")
        r     = rank_biserial_r(scores_neg, scores_pos)
        auroc = compute_auroc(scores_neg, scores_pos)
        auprc = compute_auprc(scores_neg, scores_pos)
        rows.append({
            "comparison":      label,
            "layer":           best_layer,
            "K":               best_K,
            "n_negative":      len(scores_neg),
            "n_positive":      len(scores_pos),
            "AUROC":           round(auroc, 4),
            "AUPRC":           round(auprc, 4),
            "MannWhitneyU":    float(U),
            "p_value":         float(p),
            "rank_biserial_r": round(r, 4),
        })

    df_stats = pd.DataFrame(rows)
    out_csv  = RESULTS_DIR / "stats_summary.csv"
    df_stats.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")
    print(df_stats.to_string(index=False))

    print(f"\nAll results saved to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",            default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--normative-file",   default="data/raw/normative.txt")
    p.add_argument("--harmful-file",     default="data/raw/harmful.txt")
    p.add_argument("--benign-agg-file",  default="data/raw/benign_aggressive.txt")
    p.add_argument("--normative-n",      type=int, default=200)
    p.add_argument("--harmful-n",        type=int, default=200)
    p.add_argument("--benign-agg-n",     type=int, default=200)
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
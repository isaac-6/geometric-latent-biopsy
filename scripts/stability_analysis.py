"""
stability_analysis.py
---------------------
Measures how detection performance and reference direction stability change
as a function of normative set size N.

Primary metric  — AUROC(N)
    Fit the normative reference on the first N prompts; score a fixed held-out
    set of harmful and benign-aggressive prompts; compute AUROC.
    This directly answers: "how many normative prompts do I need before
    detection performance plateaus?" — the operationally meaningful question.

Secondary metric — PC1 angle drift(N)
    Angular distance between the PC1 computed on N prompts and the PC1 on
    the full set.  Useful as a geometric sanity check but not sufficient on
    its own: a direction that has drifted 20° may still yield identical AUROC
    if the manifold has a broad safe basin.

Both metrics are computed in two orderings:
    Forward  : fit on the first N prompts (growing N).
    Reverse  : fit on the last N prompts (complementary subset, growing N).
    Agreement between orderings rules out sampling-order artefacts.

Outputs (under results/eval/):
    stability_auroc.png        AUROC vs N — primary figure (paper-ready)
    stability_pc1_angle.png    PC1 angle drift vs N — appendix figure
    stability_summary.csv      all numerical values

Usage
-----
    python scripts/stability_analysis.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --normative-file   data/raw/normative.txt \\
        --harmful-file     data/raw/harmful.txt \\
        --benign-agg-file  data/raw/benign_aggressive.txt \\
        --normative-n  500 \\
        --harmful-n    200 \\
        --benign-agg-n 200 \\
        --layers 0 6 12 19 22 \\
        --target-layer 19 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction import LatentExtractor   # type: ignore[import-untyped]
from theta import ThetaBiomarker         # type: ignore[import-untyped]

RESULTS_DIR = Path("results/eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts(path: str, n: int, seed: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(seed)
    return random.sample(lines, min(n, len(lines)))


def compute_auroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """AUROC where scores_pos should be higher for true positives."""
    y_true  = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    y_score = np.concatenate([scores_neg, scores_pos])
    if np.isnan(y_score).any() or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _normalise_pc1(v: np.ndarray) -> np.ndarray:
    """Fix PCA sign ambiguity: make first non-near-zero coefficient positive."""
    v = v.copy()
    nz = np.where(np.abs(v) > 1e-9)[0]
    if len(nz) > 0 and v[nz[0]] < 0:
        v = -v
    return v


def pc1_angle_deg(X_subset: np.ndarray, pc1_full: np.ndarray) -> float:
    """
    Angle in degrees between PC1 of X_subset and a pre-computed reference PC1.
    Both vectors are sign-normalised before computing the angle.
    """
    pca = PCA(n_components=1)
    pca.fit(X_subset)
    v = _normalise_pc1(pca.components_[0])
    cos = float(np.clip(np.dot(v, pc1_full), -1.0, 1.0))
    return float(np.degrees(np.arccos(np.abs(cos))))


def auroc_at_layer(
    norm_acts_fit:  torch.Tensor,  # (n_fit, L, D)  — used to fit the biomarker
    norm_acts_eval: torch.Tensor,  # (n_eval, L, D) — held-out normative, for scoring only
    harm_acts:      torch.Tensor,  # (N_h, L, D)
    benign_acts:    torch.Tensor,  # (N_b, L, D)
    layer: int,
) -> tuple[float, float]:
    """
    Fit a K=1 ThetaBiomarker on norm_acts_fit; score on norm_acts_eval (held-out).
    Separating fit and eval avoids the in-sample artifact where a small GMM
    memorises its training points and produces artificially perfect AUROC at
    small N.

    Returns (AUROC_harmful_vs_norm, AUROC_harmful_vs_benign).
    """
    bm = ThetaBiomarker(n_directions=1, n_gmm_components=1,
                        layer_indices=[layer])
    bm.fit(norm_acts_fit)

    # Score on held-out normative — not the fitting set
    scores_norm   = bm.score_batch(norm_acts_eval)
    scores_harm   = bm.score_batch(harm_acts)
    scores_benign = bm.score_batch(benign_acts)

    auroc_h = compute_auroc(scores_norm, scores_harm)
    auroc_b = compute_auroc(scores_benign, scores_harm)
    return auroc_h, auroc_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ---- Load prompts ----
    norm_prompts   = load_prompts(args.normative_file,  args.normative_n,  args.seed)
    harm_prompts   = load_prompts(args.harmful_file,    args.harmful_n,    args.seed)
    benign_prompts = load_prompts(args.benign_agg_file, args.benign_agg_n, args.seed)

    print(f"Normative: {len(norm_prompts)} | "
          f"Harmful: {len(harm_prompts)} | "
          f"Benign-Agg: {len(benign_prompts)}")
    print(f"Loading model: {args.model}")

    extractor = LatentExtractor(args.model)
    L = extractor.num_layers

    def extract(prompts: list[str]) -> torch.Tensor:
        return torch.stack([extractor.get_last_token_activations(p)
                            for p in prompts])

    print("Extracting normative activations...")
    norm_acts   = extract(norm_prompts)
    print("Extracting harmful activations...")
    harm_acts   = extract(harm_prompts)
    print("Extracting benign-aggressive activations...")
    benign_acts = extract(benign_prompts)

    N_norm = norm_acts.shape[0]

    # Reserve a fixed held-out normative evaluation set (20% of total).
    # The fit set grows from min_n up to N_fit_max.
    # Crucially, the eval set is NEVER used for fitting — this eliminates
    # the in-sample memorisation artifact at small N.
    n_eval    = max(20, int(0.20 * N_norm))
    n_fit_max = N_norm - n_eval
    norm_acts_eval = norm_acts[n_fit_max:]   # last n_eval prompts — fixed
    norm_acts_fit_pool = norm_acts[:n_fit_max]  # pool from which subsets are drawn

    print(f"Normative pool: {n_fit_max} fit / {n_eval} eval (held-out, fixed)")

    layers = args.layers if args.layers else list(range(L))

    # ---- Sample sizes: log-spaced over the fit pool, dense at small N ----
    min_n = max(10, int(0.02 * n_fit_max))
    sizes_raw = np.unique(
        np.round(np.geomspace(min_n, n_fit_max, num=30)).astype(int)
    )
    sizes: list[int] = [int(s) for s in sizes_raw if 2 <= s <= n_fit_max]
    if sizes[-1] != n_fit_max:
        sizes.append(n_fit_max)

    # ---- Pre-compute full-set PC1 from the entire fit pool ----
    full_pc1: dict[int, np.ndarray] = {}
    for layer in layers:
        X_full = norm_acts_fit_pool[:, layer, :].cpu().float().numpy()
        pca = PCA(n_components=1)
        pca.fit(X_full)
        full_pc1[layer] = _normalise_pc1(pca.components_[0])

    # ---- Run both orderings ----
    all_rows: list[dict] = []

    for ordering in ("forward", "reverse"):
        print(f"\nComputing {ordering} stability...")

        def subset_fn(n: int, _ord: str = ordering) -> torch.Tensor:
            # Draw from the fit pool only — eval set is always held out
            return (norm_acts_fit_pool[:n]
                    if _ord == "forward"
                    else norm_acts_fit_pool[n_fit_max - n:])

        for layer in layers:
            print(f"  Layer {layer} ...", end="", flush=True)
            for n in sizes:
                subset = subset_fn(n)
                X_sub  = subset[:, layer, :].cpu().float().numpy()

                auroc_h, auroc_b = auroc_at_layer(
                    subset, norm_acts_eval, harm_acts, benign_acts, layer
                )
                angle = (pc1_angle_deg(X_sub, full_pc1[layer])
                         if n >= 3 else float("nan"))

                all_rows.append({
                    "ordering":                ordering,
                    "layer":                   layer,
                    "n":                       n,
                    "auroc_harmful":           auroc_h,
                    "auroc_harmful_vs_benign": auroc_b,
                    "pc1_angle_deg":           angle,
                })
            print(" done")

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "stability_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    _plot_auroc_stability(df, layers, args.target_layer)
    _plot_pc1_angle(df, layers)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_auroc_stability(
    df: pd.DataFrame,
    layers: list[int],
    target_layer: int,
) -> None:
    """
    Primary stability figure. Two panels:

    Left  — AUROC(N) for harmful detection at every analysed layer (forward).
            Shows which layer converges fastest and what plateau value is.

    Right — AUROC(N) at target_layer for both comparisons (harmful-vs-norm
            and harmful-vs-benign-agg), overlaid for forward and reverse
            orderings.  Demonstrates robustness to prompt ordering.
    """
    cmap = plt.get_cmap("viridis")
    layer_colors = {l: cmap(i / max(len(layers) - 1, 1))
                    for i, l in enumerate(layers)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1 — all layers, forward, harmful-vs-norm
    ax = axes[0]
    fwd = df[df["ordering"] == "forward"]
    for layer in layers:
        sub = fwd[fwd["layer"] == layer].sort_values("n")
        ax.plot(sub["n"], sub["auroc_harmful"],
                marker="o", markersize=3, lw=1.8,
                color=layer_colors[layer], label=f"Layer {layer}")

    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="AUROC=0.90")
    ax.set_xscale("log")
    ax.set_xlabel("Normative set size (N)")
    ax.set_ylabel("AUROC — harmful vs normative")
    ax.set_title("Harmful detection AUROC vs N\n(forward ordering, all layers)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)

    # Panel 2 — target layer, both comparisons, forward + reverse
    ax = axes[1]
    style_map = {
        ("forward",  "auroc_harmful"):           ("#d62728", "-",  "Harm/norm (forward)"),
        ("forward",  "auroc_harmful_vs_benign"): ("#2ca02c", "-",  "Harm/benign (forward)"),
        ("reverse",  "auroc_harmful"):           ("#d62728", "--", "Harm/norm (reverse)"),
        ("reverse",  "auroc_harmful_vs_benign"): ("#2ca02c", "--", "Harm/benign (reverse)"),
    }
    for (ordering, col), (color, ls, label) in style_map.items():
        sub = df[(df["ordering"] == ordering) &
                 (df["layer"] == target_layer)].sort_values("n")
        ax.plot(sub["n"], sub[col],
                color=color, ls=ls, marker="o", markersize=3,
                lw=1.8, label=label, alpha=0.9)

    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Normative set size (N)")
    ax.set_ylabel("AUROC")
    ax.set_title(
        f"AUROC vs N — Layer {target_layer}\n"
        "Forward vs reverse ordering (ordering robustness)"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "stability_auroc.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def _plot_pc1_angle(df: pd.DataFrame, layers: list[int]) -> None:
    """
    Appendix figure: PC1 direction drift vs N for forward and reverse orderings.
    """
    cmap = plt.get_cmap("viridis")
    layer_colors = {l: cmap(i / max(len(layers) - 1, 1))
                    for i, l in enumerate(layers)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, ordering, suffix in zip(
        axes,
        ["forward", "reverse"],
        ["forward: growing N", "reverse: shrinking N"],
    ):
        sub = df[df["ordering"] == ordering]
        for layer in layers:
            row = sub[sub["layer"] == layer].sort_values("n")
            ax.plot(row["n"], row["pc1_angle_deg"],
                    marker="o", markersize=3, lw=1.8,
                    color=layer_colors[layer], label=f"Layer {layer}")

        ax.axhline(5.0, color="gray", linestyle="--", alpha=0.6,
                   label="5° threshold")
        ax.set_xscale("log")
        ax.set_xlabel("Normative set size (N)")
        ax.set_ylabel("Angle to full-set PC1 (degrees)")
        ax.set_title(f"PC1 direction drift — {suffix}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "stability_pc1_angle.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Normative set size stability analysis."
    )
    p.add_argument("--model",           default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--normative-file",  default="data/raw/normative.txt")
    p.add_argument("--harmful-file",    default="data/raw/harmful.txt")
    p.add_argument("--benign-agg-file", default="data/raw/benign_aggressive.txt")
    p.add_argument("--normative-n",     type=int, default=500)
    p.add_argument("--harmful-n",       type=int, default=200)
    p.add_argument("--benign-agg-n",    type=int, default=200)
    p.add_argument("--layers",          type=int, nargs="+", default=None,
                   help="Layers to analyse. Default: all. "
                        "Recommended: --layers 0 6 12 19 22")
    p.add_argument("--target-layer",    type=int, default=19,
                   help="Layer shown in the two-ordering robustness panel.")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
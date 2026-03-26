"""
plot_theta_phi_full.py
----------------------
Generates the theta-phi orthogonal projection plane using the full evaluation
datasets (AdvBench, XSTest, Alpaca-normative) rather than hand-crafted prompts.

What this plot shows
--------------------
Each prompt is represented as a point in polar coordinates (θ, φ) where:
    θ (theta) = angle between the prompt's activation and the normative PC1
                direction — the radial anomaly score.
    φ (phi)   = azimuthal angle in the orthogonal complement of PC1, found
                via 2D PCA of the residual vectors.  It encodes the *direction*
                of deviation, not only its magnitude.

The plot is drawn in Cartesian coordinates (θ·cos φ, θ·sin φ) so that:
    — Distance from the origin = θ (directly readable from concentric circles)
    — Angle from the x-axis = φ

Four categories are shown:
    Normative-train  (blue filled)   — prompts used to fit PC1
    Normative-test   (blue hollow)   — held-out normative prompts
    Harmful          (red ×)         — AdvBench
    Benign-aggressive (green △)      — XSTest

Plotting normative-train vs normative-test lets us verify that the reference
direction does not overfit to the training subset.

Design decisions
----------------
- The normative PC1 is fitted on normative-train only.
- The target layer defaults to 19 (empirically best from evaluate_biomarker).
- A random sample of each class is shown when the full set is large, to keep
  the plot legible.  Sample size is configurable.

Outputs (under results/figures/):
    theta_phi_full_layer{L}.png

Usage
-----
    python scripts/plot_theta_phi_full.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --normative-file   data/raw/normative.txt \\
        --harmful-file     data/raw/harmful.txt \\
        --benign-agg-file  data/raw/benign_aggressive.txt \\
        --normative-n      200 \\
        --harmful-n        100 \\
        --benign-agg-n     100 \\
        --train-fraction   0.6 \\
        --layer            19 \\
        --seed             42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction import LatentExtractor   # type: ignore[import-untyped]
from theta import compute_theta_core     # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prompts(path: str, n: int, seed: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(seed)
    return random.sample(lines, min(n, len(lines)))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _perp(X: torch.Tensor, pc1_unit: torch.Tensor) -> torch.Tensor:
    """Orthogonal rejection of X from pc1_unit: X_perp = X - (X·c)c."""
    dot = (X * pc1_unit).sum(dim=-1, keepdim=True)
    return X - dot * pc1_unit


def fit_phi_basis(
    X_fit: torch.Tensor,   # (N_fit, D) — normative TRAINING activations only
    pc1:   torch.Tensor,   # (D,) — reference direction (unit vector)
) -> "PCA":
    """
    Fit the 2D phi basis from the normative TRAINING set only.
    This ensures phi is a one-shot coordinate: defined entirely from safe data,
    applicable to any unseen prompt without re-fitting.

    Returns a fitted sklearn PCA(n_components=2) object.
    """
    pc1_unit = pc1 / pc1.norm()
    X_fit_perp = _perp(X_fit, pc1_unit)
    phi_pca = PCA(n_components=2)
    phi_pca.fit(X_fit_perp.cpu().float().numpy())
    return phi_pca


def compute_theta_phi(
    X_all:    torch.Tensor,  # (N_total, D) — all points to project
    pc1:      torch.Tensor,  # (D,) — reference direction (unit vector)
    phi_pca:  "PCA",         # fitted PCA from fit_phi_basis (normative train only)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project all points onto the (θ, φ) coordinate system.

    θ is measured from the reference direction pc1.
    φ is the azimuthal angle in the 2D plane defined by phi_pca,
    which was fit on the normative training set only — making both
    coordinates fully one-shot (no test data used to define the basis).

    Returns
    -------
    theta : (N_total,) float array — angle from pc1 in [0, π]
    phi   : (N_total,) float array — azimuthal angle in the normative
            orthogonal plane [-π, π]
    """
    pc1_unit = pc1 / pc1.norm()
    ref      = pc1_unit.unsqueeze(0)

    # θ: angle between each activation and the reference direction
    theta = compute_theta_core(X_all, ref.expand(X_all.shape[0], -1))  # (N,)

    # φ: project orthogonal components onto the pre-fit normative PCA basis
    X_all_perp = _perp(X_all, pc1_unit)                                # (N, D)
    X_perp_2d  = phi_pca.transform(X_all_perp.cpu().float().numpy())   # (N, 2)
    phi        = np.arctan2(X_perp_2d[:, 1], X_perp_2d[:, 0])         # (N,)

    return theta.cpu().numpy(), phi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ---- Load prompts ----
    norm_prompts   = load_prompts(args.normative_file,  args.normative_n,  args.seed)
    harm_prompts   = load_prompts(args.harmful_file,    args.harmful_n,    args.seed)
    benign_prompts = load_prompts(args.benign_agg_file, args.benign_agg_n, args.seed)

    # Absolute fit-N from stability analysis — maximises held-out eval set
    n_norm_fit = min(args.normative_fit_n, len(norm_prompts) - 1)
    n_harm_fit = min(args.harmful_fit_n,   len(harm_prompts) - 1)
    norm_train = norm_prompts[:n_norm_fit];  norm_test = norm_prompts[n_norm_fit:]
    harm_train = harm_prompts[:n_harm_fit];  harm_test = harm_prompts[n_harm_fit:]

    print(f"Normative: {len(norm_train)} fit / {len(norm_test)} test")
    print(f"Harmful:   {len(harm_train)} fit / {len(harm_test)} test  |  "
          f"Benign-Agg: {len(benign_prompts)}")

    print(f"\nLoading model: {args.model}")
    extractor = LatentExtractor(args.model)
    layer = args.layer

    def extract_layer(prompts: list[str]) -> torch.Tensor:
        acts = torch.stack([extractor.get_last_token_activations(p)
                            for p in prompts])
        return acts[:, layer, :]

    print("Extracting activations...")
    X_norm_train = extract_layer(norm_train)
    X_norm_test  = extract_layer(norm_test)
    X_harm_train = extract_layer(harm_train)
    X_harm_test  = extract_layer(harm_test)
    X_benign     = extract_layer(benign_prompts)

    strategies = (["normative_ref", "harmful_ref"]
                  if args.strategy == "both" else [args.strategy])

    for strategy in strategies:
        print(f"\n=== Strategy: {strategy} ===")
        X_fit     = X_norm_train if strategy == "normative_ref" else X_harm_train
        ref_label = "Normative PC1" if strategy == "normative_ref" else "Harmful PC1"

        pca_ref = PCA(n_components=1)
        pca_ref.fit(X_fit.cpu().float().numpy())
        pc1_vec = torch.tensor(pca_ref.components_[0],
                               dtype=X_fit.dtype, device=X_fit.device)

        # Fit phi basis on the FIT set only (one-shot: no test data used)
        phi_pca = fit_phi_basis(X_fit, pc1_vec)

        # Setup data splits based on strategy
        if strategy == "normative_ref":
            X_harm_all = torch.cat([X_harm_train, X_harm_test], dim=0)
            parts    =[X_norm_train, X_norm_test, X_harm_all, X_benign]
            cat_names =["norm_train", "norm_test", "harm", "benign"]
            # Keep flat list of prompt text perfectly aligned with tensors
            parts_prompts = norm_train + norm_test + harm_train + harm_test + benign_prompts
        else:
            parts    =[X_norm_train, X_norm_test, X_harm_train, X_harm_test, X_benign]
            cat_names =["norm_train", "norm_test", "harm_train", "harm_test", "benign"]
            parts_prompts = norm_train + norm_test + harm_train + harm_test + benign_prompts

        sizes = [x.shape[0] for x in parts]
        X_all = torch.cat(parts, dim=0)

        print("Computing theta-phi coordinates (phi basis from fit set only)...")
        theta_all, phi_all = compute_theta_phi(X_all, pc1_vec, phi_pca)

        idx = np.cumsum([0] + sizes)
        theta_by_cat = {n: theta_all[idx[i]:idx[i+1]] for i, n in enumerate(cat_names)}
        phi_by_cat   = {n: phi_all[idx[i]:idx[i+1]]   for i, n in enumerate(cat_names)}

        # 1. Generate the visual plot
        _plot(theta_by_cat, phi_by_cat, layer, strategy, ref_label, figures_dir, args)

        # 2. Export CSV for R (ggplot2) if flag is passed
        if args.export_csv:
            import pandas as pd
            from theta import ThetaBiomarker
            
            print("Exporting raw scores to CSV...")
            
            # Trick the biomarker into processing our (N, D) tensor natively
            bm = ThetaBiomarker(n_directions=1, n_gmm_components=1, layer_indices=[0])
            bm.fit(X_fit.unsqueeze(1))
            scores_all = bm.score_batch(X_all.unsqueeze(1))
            
            # Apply exact same scoring logic as evaluate_biomarker to guarantee parity
            if strategy == "harmful_ref":
                scores_all = -scores_all
                idx_norm = cat_names.index("norm_test")
                idx_harm = cat_names.index("harm_test")
                s_norm = scores_all[idx[idx_norm]:idx[idx_norm+1]]
                s_harm = scores_all[idx[idx_harm]:idx[idx_harm+1]]
                if float(np.mean(s_harm)) < float(np.mean(s_norm)):
                    scores_all = -scores_all
            
            flat_labels = np.repeat(cat_names, sizes)
            
            df = pd.DataFrame({
                "label": flat_labels,
                "theta": np.round(theta_all, 5),
                "phi": np.round(phi_all, 5),
                "anomaly_score": np.round(scores_all, 5),
                "prompt": parts_prompts
            })
            
            out_csv = figures_dir / f"scores_{strategy}_layer{layer}.csv"
            df.to_csv(out_csv, index=False, encoding="utf-8")
            print(f"Saved CSV -> {out_csv}")

def _plot(
    theta:       dict[str, np.ndarray],
    phi:         dict[str, np.ndarray],
    layer:       int,
    strategy:    str,
    ref_label:   str,
    figures_dir: Path,
    args:        argparse.Namespace,
) -> None:
    """
    Render the theta-phi plane.  Each point is at (θ·cos φ, θ·sin φ).

    For normative_ref: all harmful prompts shown as a single class (none used
    for fitting).  For harmful_ref: harmful split into train/test to verify
    that the PC1 direction generalises to held-out harmful examples.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    def cart(cat: str) -> tuple[np.ndarray, np.ndarray]:
        t, p = theta[cat], phi[cat]
        return t * np.cos(p), t * np.sin(p)

    # ---- Normative (always shown as train / test split) ----
    px, py = cart("norm_train")
    ax.scatter(px, py, c="#1f77b4", s=50, alpha=0.5, marker="o",
               label=f"Normative fit  (n={len(px)})")
    px, py = cart("norm_test")
    ax.scatter(px, py, c="#1f77b4", s=55, alpha=0.9, marker="o",
               edgecolors="white", linewidths=0.9,
               label=f"Normative eval (n={len(px)})")

    # ---- Benign-aggressive ----
    px, py = cart("benign")
    ax.scatter(px, py, c="#2ca02c", s=55, alpha=0.7, marker="^",
               label=f"Benign-agg / XSTest (n={len(px)})")

    # ---- Harmful: single class for normative_ref, split for harmful_ref ----
    if "harm" in theta:
        # normative_ref: all 520 harmful prompts are eval data
        px, py = cart("harm")
        ax.scatter(px, py, c="#d62728", s=65, alpha=0.8, marker="X",
                   label=f"Harmful / AdvBench (n={len(px)})")
    else:
        # harmful_ref: show fit vs held-out to verify generalisation
        px, py = cart("harm_train")
        ax.scatter(px, py, c="#d62728", s=60, alpha=0.4, marker="X",
                   label=f"Harmful fit  (n={len(px)})")
        px, py = cart("harm_test")
        ax.scatter(px, py, c="#d62728", s=70, alpha=0.9, marker="X",
                   edgecolors="white", linewidths=0.9,
                   label=f"Harmful eval (n={len(px)})")

    # ---- Reference direction marker ----
    ax.plot(0, 0, marker="*", markersize=18, color="black",
            zorder=10, label=ref_label)

    # ---- Concentric theta circles — sparse labels at 50° to avoid overlap ----
    all_theta = np.concatenate(list(theta.values()))
    max_theta = float(np.max(all_theta))

    # Draw a thin circle at every 0.25 rad; label only every 0.5 rad
    label_angle_rad = np.radians(50)   # upper-right diagonal — away from data
    cos_a, sin_a = np.cos(label_angle_rad), np.sin(label_angle_rad)

    for r in np.arange(0.25, max_theta + 0.25, 0.25):
        circle = mpatches.Circle((0, 0), r, color="gray", fill=False,
                                  linestyle="--", alpha=0.30, linewidth=0.7)
        ax.add_patch(circle)
        # Only label multiples of 0.5
        if abs(round(r * 2) - r * 2) < 1e-9:   # r is a multiple of 0.5
            ax.text(r * cos_a + 0.03, r * sin_a + 0.03,
                    f"θ = {r:.1f}",
                    color="gray", fontsize=8, ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.6, pad=0.5))

    # ---- Axes and formatting ----
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.25)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.25)
    lim = max_theta * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Theta-Phi Projection — Layer {layer}  [{strategy}]\n"
        f"Reference: {ref_label}  |  Alpaca / AdvBench / XSTest",
        fontsize=13,
    )
    # Axis labels: θ cos φ and θ sin φ are the correct Cartesian coordinates
    # of the polar representation (θ = angular deviation, φ = azimuthal angle
    # in the 2D orthogonal complement of the reference direction).
    ax.set_xlabel(r"$\theta \cos \varphi$", fontsize=13)
    ax.set_ylabel(r"$\theta \sin \varphi$", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    out = str(figures_dir / f"theta_phi_{strategy}_layer{layer}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")

    # ---- Summary statistics ----
    print(f"  {'Category':<14}  {'mean':>6}  {'median':>6}  {'std':>6}  n")
    for cat, t in theta.items():
        print(f"  {cat:<14}  {t.mean():>6.3f}  {float(np.median(t)):>6.3f}  "
              f"{t.std():>6.3f}  {len(t)}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Theta-phi projection for both reference strategies."
    )
    p.add_argument("--model",           default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--normative-file",  default="data/raw/normative.txt")
    p.add_argument("--harmful-file",    default="data/raw/harmful.txt")
    p.add_argument("--benign-agg-file", default="data/raw/benign_aggressive.txt")
    p.add_argument("--normative-n",     type=int, default=500)
    p.add_argument("--harmful-n",       type=int, default=520)
    p.add_argument("--benign-agg-n",    type=int, default=250)
    p.add_argument("--normative-fit-n", type=int, default=200,
                   help="Prompts used to fit normative PC1 (stability-analysis justified).")
    p.add_argument("--harmful-fit-n",   type=int, default=200,
                   help="Prompts used to fit harmful PC1 (harmful_ref only).")
    p.add_argument("--layer",           type=int, default=19)
    p.add_argument("--strategy",        default="normative_ref",
                   choices=["normative_ref", "harmful_ref", "both"])
    p.add_argument("--figures-dir",     default="results/figures",
                   help="Output directory for figures.")
    p.add_argument("--export-csv", action="store_true",
                   help="Export raw scores and text to CSV.")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
"""
stability_analysis.py
---------------------
Measures how stable the normative reference direction (PC1) is as a function
of the number of prompts used to fit it.

Two complementary views:
  1. Forward (growing):  fit on N prompts, measure angular distance to the
     direction computed on the full set.  Shows convergence rate.
  2. Reverse (shrinking): fit on full set, progressively remove prompts and
     measure drift.  Confirms early stability is not an artefact of ordering.

Both curves should plateau well before the full N, justifying the chosen
normative set size.

Outputs (under results/eval/):
    stability_forward.png   — PC1 angle vs full-set PC1 as N grows
    stability_reverse.png   — PC1 angle vs full-set PC1 as N shrinks
    stability_summary.csv   — numerical values for both curves at each layer

Usage
-----
    python scripts/stability_analysis.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --normative-file data/raw/normative.txt \
        --normative-n 500 \
        --layers 0 6 12 19 22 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction import LatentExtractor          # type: ignore[import-untyped]

RESULTS_DIR = Path("results/eval")


# ---------------------------------------------------------------------------
# Core: PC1 stability metric
# ---------------------------------------------------------------------------

def pc1_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    """
    Angle in degrees between PC1 of two activation matrices A and B.
    PC1 direction is sign-normalised (flip so first non-zero element > 0)
    to remove the arbitrary sign ambiguity of PCA.
    """
    def _pc1(X: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=1)
        pca.fit(X)
        v = pca.components_[0]
        # sign normalisation: make the first non-near-zero component positive
        nz = np.where(np.abs(v) > 1e-9)[0]
        if len(nz) > 0 and v[nz[0]] < 0:
            v = -v
        return v

    v_a = _pc1(A)
    v_b = _pc1(B)
    cos = np.clip(np.dot(v_a, v_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(np.abs(cos))))  # abs: sign-agnostic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ---- Load prompts ----
    with open(args.normative_file, encoding="utf-8") as f:
        all_prompts = [l.strip() for l in f if l.strip()]

    n_total = min(args.normative_n, len(all_prompts))
    prompts = random.sample(all_prompts, n_total)
    print(f"Using {n_total} normative prompts.")

    # ---- Extract activations ----
    print(f"Loading model: {args.model}")
    extractor = LatentExtractor(args.model)

    print("Extracting activations...")
    acts = torch.stack([
        extractor.get_last_token_activations(p) for p in prompts
    ])  # (N, L, D)
    acts_np = acts.cpu().float().numpy()
    N, L, D = acts_np.shape
    print(f"  Shape: {acts_np.shape}")

    # ---- Choose layers ----
    layers = args.layers if args.layers else list(range(L))

    # ---- Sample sizes to evaluate ----
    # Dense at small N (where convergence happens), sparse at large N
    min_n = max(8, int(0.02 * N))  # need at least 2 samples for PCA
    sizes_raw = np.unique(np.round(
        np.geomspace(min_n, N, num=30)
    ).astype(int))
    sizes: list[int] = [int(s) for s in sizes_raw if s <= N]
    if sizes[-1] != N:
        sizes.append(N)

    # ---- Forward stability ----
    print("\nComputing forward stability (N → full)...")
    # Reference: PC1 on the full set
    forward_rows = []
    for layer in layers:
        X_full = acts_np[:, layer, :]
        for n in sizes:
            # Use the first n prompts (fixed order → reproducible)
            angle = pc1_angle_deg(acts_np[:n, layer, :], X_full)
            forward_rows.append({"curve": "forward", "layer": layer,
                                  "n": n, "angle_deg": angle})
        print(f"  Layer {layer}: done")

    # ---- Reverse stability ----
    print("\nComputing reverse stability (full → N)...")
    # Reference: same full-set PC1. Remove prompts in reverse order.
    reverse_rows = []
    for layer in layers:
        X_full = acts_np[:, layer, :]
        for n in sizes:
            # Take last n prompts — different subset from forward
            angle = pc1_angle_deg(acts_np[N - n:, layer, :], X_full)
            reverse_rows.append({"curve": "reverse", "layer": layer,
                                  "n": n, "angle_deg": angle})
        print(f"  Layer {layer}: done")

    df = pd.DataFrame(forward_rows + reverse_rows)
    out_csv = RESULTS_DIR / "stability_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ---- Plot ----
    _plot_stability(df, layers, sizes, "forward",
                    RESULTS_DIR / "stability_forward.png",
                    "PC1 angle vs full-set direction (forward: growing N)")
    _plot_stability(df, layers, sizes, "reverse",
                    RESULTS_DIR / "stability_reverse.png",
                    "PC1 angle vs full-set direction (reverse: shrinking N)")


def _plot_stability(df: pd.DataFrame, layers: list[int], sizes: list[int],
                    curve: str, out: Path, title: str):
    sub = df[df["curve"] == curve]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(layers) - 1, 1)) for i in range(len(layers))]

    for layer, c in zip(layers, colors):
        row = sub[sub["layer"] == layer].sort_values("n")
        ax.plot(row["n"], row["angle_deg"], marker="o", markersize=4,
                label=f"Layer {layer}", color=c, lw=1.8)

    ax.axhline(5.0, color="gray", linestyle="--", alpha=0.6,
               label="5° stability threshold")
    ax.set_xlabel("Normative set size (N)")
    ax.set_ylabel("Angle to full-set PC1 (degrees)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--normative-file",  default="data/raw/normative.txt")
    p.add_argument("--normative-n",     type=int, default=500)
    p.add_argument("--layers",          type=int, nargs="+", default=None,
                   help="Layers to analyse (default: all). E.g. --layers 0 6 12 19 22")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
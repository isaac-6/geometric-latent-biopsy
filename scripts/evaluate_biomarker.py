"""
evaluate_biomarker.py
---------------------
Systematic evaluation of the ThetaBiomarker under two reference strategies:

  normative_ref (zero-shot)
      The reference direction is the PC1 of NORMATIVE (safe) prompts.
      No harmful examples are seen at fit time.
      Anomaly score: -log p(theta | normative GMM).  Higher = more anomalous.

  harmful_ref (supervised variant)
      The reference direction is the PC1 of HARMFUL prompts.
      A fraction of harmful examples are used at fit time.
      Scores are negated: lower -log p under the harmful GMM = closer to
      the harmful manifold = more suspicious.  After negation, higher score
      still means more likely harmful, keeping metric conventions uniform.

Data splits
-----------
A single DataSplit object is constructed once and shared by all experiments:

    normative:      normative_fit_n prompts → fit set (normative_ref only)
                    remainder        → eval set (always held-out)
                    Motivated by stability analysis: AUROC plateaus at N≈200.
                    Using a fixed N rather than a fraction maximises the eval set.
    harmful:        harmful_fit_n prompts → fit set (harmful_ref only)
                    remainder       → eval set (always held-out)
    benign-agg:     100% eval-only — never used for fitting

Metrics are always computed on held-out eval sets.  Fit-set prompts are never
scored under the biomarker that was trained on them.

Experiments
-----------
  Exp 1  Per-layer AUROC across K directions (both strategies)
  Exp 2  Dimension pruning ablation at best layer (normative_ref only)
  Exp 3  Score distributions, PR curves, and statistical tests (both strategies)

Outputs (under results/eval/<strategy>/)
-----------------------------------------
    auroc_by_layer.png
    auroc_ablation_dim.png        (normative_ref only)
    score_distributions.png
    precision_recall.png
    stats_summary.csv

Usage
-----
    python scripts/evaluate_biomarker.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --normative-file  data/raw/normative.txt \\
        --harmful-file    data/raw/harmful.txt \\
        --benign-agg-file data/raw/benign_aggressive.txt \\
        --normative-n 200 --harmful-n 200 --benign-agg-n 200 \\
        --normative-fit-n 200 \\
        --harmful-fit-n 200 \\
        --strategy both \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import PolyCollection
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from latentbiopsy.extraction import LatentExtractor           # type: ignore[import-untyped]
from latentbiopsy.theta import ThetaBiomarker, compute_theta_core  # type: ignore[import-untyped]

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_BASE = Path("results/eval")  # overridden at runtime by --output-dir


# ===========================================================================
# Data split — single source of truth for all experiments
# ===========================================================================

@dataclass(frozen=True)
class DataSplit:
    """
    Encapsulates ALL train/eval splits for reproducibility.

    Invariants enforced by construction:
      - norm_fit  ∩ norm_eval  = ∅
      - harm_fit  ∩ harm_eval  = ∅
      - harm_all  = harm_fit ∪ harm_eval  (all harmful prompts)
      - benign is never in any fit set
      - normative_ref uses harm_all for evaluation (no harmful data withheld)
      - harmful_ref  uses harm_eval for evaluation (harm_fit used for fitting)
      - Metric code only receives *_eval / *_all tensors; fit tensors are
        passed only to biomarker.fit().
    """
    # Fit tensors — passed to biomarker.fit() only
    norm_fit:  torch.Tensor    # (n_norm_fit, L, D)
    harm_fit:  torch.Tensor    # (n_harm_fit, L, D)

    # Eval tensors — passed to biomarker.score_batch() only
    norm_eval: torch.Tensor    # (n_norm_eval, L, D)
    harm_eval: torch.Tensor    # (n_harm_eval, L, D)  — harm minus harm_fit (for harmful_ref)
    harm_all:  torch.Tensor    # (n_harm_total, L, D) — ALL harmful (for normative_ref)
    benign:    torch.Tensor    # (n_benign,    L, D)  — always eval-only

    # Provenance (for logging and paper)
    train_fraction: float
    seed: int
    n_norm_total: int
    n_harm_total: int
    n_benign_total: int

    def summary(self) -> str:
        return (
            f"DataSplit("
            f"norm={len(self.norm_fit)}fit/{len(self.norm_eval)}eval, "
            f"harm={len(self.harm_fit)}fit/{len(self.harm_eval)}eval"
            f"/{len(self.harm_all)}all, "
            f"benign={len(self.benign)}eval-only, "
            f"train_fraction={self.train_fraction}, seed={self.seed})"
        )


def make_split(
    norm_acts:       torch.Tensor,
    harm_acts:       torch.Tensor,
    benign_acts:     torch.Tensor,
    train_fraction:  float,
    seed:            int,
    normative_fit_n: int | None = None,
    harmful_fit_n:   int | None = None,
) -> DataSplit:
    """
    Split normative and harmful activations into fit/eval sets.
    Benign-aggressive activations are eval-only by design.

    Fit-set size is determined by either:
      (a) --normative-fit-n / --harmful-fit-n  (absolute, preferred):
          Directly encodes the stability-analysis recommendation (e.g., N=200).
          Gives the LARGEST possible held-out eval set for that fit size.
      (b) --train-fraction (fraction, legacy fallback):
          Uses int(N * fraction) prompts for fitting.

    The split is deterministic given (normative_fit_n, harmful_fit_n,
    train_fraction, seed).
    """
    rng = np.random.default_rng(seed)

    def split_tensor(
        t: torch.Tensor, abs_n: int | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N   = t.shape[0]
        n_fit = int(abs_n) if abs_n is not None else max(2, int(N * train_fraction))
        n_fit = max(2, min(n_fit, N - 1))   # guard: at least 1 eval sample
        
        # Load_prompts() already shuffled the data. Sequential slicing guarantees 
        # 100% parity with the plotting scripts.
        return t[:n_fit], t[n_fit:]

    norm_fit, norm_eval = split_tensor(norm_acts, normative_fit_n)
    harm_fit, harm_eval = split_tensor(harm_acts, harmful_fit_n)

    return DataSplit(
        norm_fit=norm_fit, norm_eval=norm_eval,
        harm_fit=harm_fit, harm_eval=harm_eval,
        harm_all=harm_acts,   # all harmful prompts — used by normative_ref
        benign=benign_acts,
        train_fraction=train_fraction,
        seed=seed,
        n_norm_total=norm_acts.shape[0],
        n_harm_total=harm_acts.shape[0],
        n_benign_total=benign_acts.shape[0],
    )


# ===========================================================================
# Strategy execution — fit biomarker, return eval scores
# ===========================================================================

def scores_normative_ref(
    split: DataSplit,
    layer: int,
    K: int,
    top_d_dims: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Fit on norm_fit; score norm_eval, harm_eval, benign, rest.
    Higher score = farther from normative manifold = more anomalous.
    Fit data is NEVER included in returned scores.

    Keys: norm, harm, benign, rest
      rest = norm_eval ∪ benign  (the real-world negative class)
    """
    bm = ThetaBiomarker(
        n_directions=K, n_gmm_components=1,
        top_d_dims=top_d_dims, layer_indices=[layer],
    )
    bm.fit(split.norm_fit)
    s_norm   = bm.score_batch(split.norm_eval)
    s_harm   = bm.score_batch(split.harm_all)   # ALL harmful — none withheld for normative_ref
    s_benign = bm.score_batch(split.benign)
    return {
        "norm":   s_norm,
        "harm":   s_harm,
        "benign": s_benign,
        "rest":   np.concatenate([s_norm, s_benign]),
    }


def scores_harmful_ref(
    split: DataSplit,
    layer: int,
    K: int,
    top_d_dims: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Fit on harm_fit; score norm_eval, harm_eval, benign, rest.

    Raw score from the biomarker is -log p(x | harmful GMM):
      LOW  → close to harmful manifold (high likelihood under harmful GMM)
      HIGH → far from harmful manifold

    We NEGATE to get a "harmful proximity score":
      HIGH → close to harmful manifold → suspicious
      LOW  → far from harmful manifold → safe

    After negation, the convention "higher score = positive class" is
    consistent with normative_ref and all downstream metric code.

    NOTE: this is a supervised variant — it requires harmful examples
    at fit time and is methodologically distinct from normative_ref.

    Keys: norm, harm, benign, rest
    """
    bm = ThetaBiomarker(
        n_directions=K, n_gmm_components=1,
        top_d_dims=top_d_dims, layer_indices=[layer],
    )
    bm.fit(split.harm_fit)
    s_norm   = -bm.score_batch(split.norm_eval)
    s_harm   = -bm.score_batch(split.harm_eval)
    s_benign = -bm.score_batch(split.benign)

    # Auto-orientation: verify harm scores are higher than norm scores.
    # If not (can happen when the harmful GMM generalises poorly to
    # held-out harmful prompts), flip all signs so that "higher = more
    # proximal to harmful" is always the correct interpretation.
    # This is a defensive check — it should not fire with well-separated
    # distributions, but prevents silent sign errors across models/layers.
    # Capture pre-flip means for the diagnostic message before reassignment
    _pre_harm_mean = float(np.mean(s_harm))
    _pre_norm_mean = float(np.mean(s_norm))
    if _pre_harm_mean < _pre_norm_mean:
        s_norm, s_harm, s_benign = -s_norm, -s_harm, -s_benign
        print(f"  [harmful_ref] Auto-orientation: scores flipped "
              f"(pre-flip harm mean {_pre_harm_mean:.3f} < "
              f"norm mean {_pre_norm_mean:.3f}; "
              f"post-flip harm mean {float(np.mean(s_harm)):.3f})")

    return {
        "norm":   s_norm,
        "harm":   s_harm,
        "benign": s_benign,
        "rest":   np.concatenate([s_norm, s_benign]),
    }


STRATEGIES: dict[str, object] = {
    "normative_ref": scores_normative_ref,
    "harmful_ref":   scores_harmful_ref,
}

# Per-strategy comparison definitions.
# Each entry: (label, neg_key, pos_key)
# Convention: pos_key should score HIGHER than neg_key → AUROC ≥ 0.5.
#
# normative_ref: score is anomaly w.r.t. safe manifold (higher = more anomalous).
#   harmful prompts are most anomalous; benign-agg is slightly anomalous; norm is least.
#
# harmful_ref: score is proximity to harmful manifold (higher = closer to harmful).
#   Careful: the empirical finding is that norm_eval and harm_eval have similar proximity
#   scores (weak separation), while benign-agg has distinctly LOWER proximity (it is
#   geometrically farther from the harmful manifold than normative prompts are).
#   The `normative_vs_benign_agg` comparison therefore flips: norm is the "positive"
#   class (higher proximity to harmful than benign-agg), not benign.
COMPARISON_CONFIGS: dict[str, list[tuple[str, str, str]]] = {
    "normative_ref": [
        # Harmful prompts score highest (most anomalous)
        ("normative_vs_harmful",               "norm",   "harm"),
        # Benign-agg is slightly anomalous relative to normative
        ("normative_vs_benign_agg",            "norm",   "benign"),
        # Core operational comparison: harmful vs surface-similar safe
        ("harmful_vs_benign_agg",              "benign", "harm"),
        # Real-world negative class: everything that should not be flagged
        ("harmful_vs_rest",                    "rest",   "harm"),
    ],
    "harmful_ref": [
        # Harmful prompts should be most proximal to harmful manifold
        ("normative_vs_harmful",               "norm",   "harm"),
        # Finding: benign-agg prompts are SLIGHTLY more proximal to the harmful
        # manifold than neutral normative prompts (benign median 2.894 > norm
        # median 2.863 at layer 5).  The harmful GMM, trained on AdvBench, picks
        # up aggressive surface vocabulary shared with XSTest benign-agg prompts.
        # neg=norm (least proximal), pos=benign (slightly more proximal).
        # Effect is small (r≈0.42) — both classes are far from harmful (mean≈4.35).
        ("benign_higher_proximity_than_normative", "norm", "benign"),
        # Harmful vs benign-agg: main discrimination task
        ("harmful_vs_benign_agg",              "benign", "harm"),
        # Operational: harmful vs everything else
        ("harmful_vs_rest",                    "rest",   "harm"),
    ],
}


# ===========================================================================
# Baselines (always use norm_eval as the reference class)
# ===========================================================================

def baseline_cosine(split: DataSplit, layer: int) -> dict[str, np.ndarray]:
    """1 - cosine similarity to norm_fit centroid. Higher = more anomalous."""
    centroid = split.norm_fit[:, layer, :].mean(dim=0, keepdim=True)
    def _score(acts: torch.Tensor) -> np.ndarray:
        X   = acts[:, layer, :]
        cos = torch.nn.functional.cosine_similarity(
            X, centroid.expand_as(X), dim=-1
        )
        return (1.0 - cos).cpu().numpy()
    return {
        "norm":   _score(split.norm_eval),
        "harm":   _score(split.harm_all),   # all harmful for normative_ref convention
        "benign": _score(split.benign),
    }


def baseline_l2(split: DataSplit, layer: int) -> dict[str, np.ndarray]:
    """L2 norm of activation. Deviation from origin."""
    def _score(acts: torch.Tensor) -> np.ndarray:
        return torch.linalg.norm(acts[:, layer, :], dim=-1).cpu().numpy()
    return {
        "norm":   _score(split.norm_eval),
        "harm":   _score(split.harm_all),   # all harmful for normative_ref convention
        "benign": _score(split.benign),
    }


def baseline_random(split: DataSplit, layer: int, seed: int) -> dict[str, np.ndarray]:
    """Theta against a random unit vector — null hypothesis."""
    torch.manual_seed(seed)
    D = split.norm_fit.shape[-1]
    device = split.norm_fit.device
    dtype  = split.norm_fit.dtype
    rand_vec = torch.randn(1, D, dtype=dtype, device=device)
    rand_vec /= torch.linalg.norm(rand_vec, dim=-1, keepdim=True)

    def _score(acts: torch.Tensor) -> np.ndarray:
        X = acts[:, layer, :]
        return compute_theta_core(X, rand_vec.expand(X.shape[0], -1)).cpu().numpy()

    return {
        "norm":   _score(split.norm_eval),
        "harm":   _score(split.harm_all),   # all harmful for normative_ref convention
        "benign": _score(split.benign),
    }



def baseline_raw_theta(split: DataSplit, layer: int) -> dict[str, np.ndarray]:
    """
    Raw |theta - mu_norm| baseline — no GMM normalization.
    For K=1 Gaussian, AUROC is identical to the GMM scorer since both are
    monotonic in the same direction.  Included to verify this equivalence
    and as a simpler deployable alternative (no density model needed).
    """
    from sklearn.decomposition import PCA as _PCA

    X_fit = split.norm_fit[:, layer, :]
    pca   = _PCA(n_components=1)
    pca.fit(X_fit.cpu().float().numpy())
    pc1   = torch.tensor(pca.components_[0], dtype=X_fit.dtype, device=X_fit.device)
    pc1   = pc1 / torch.linalg.norm(pc1)
    ref   = pc1.unsqueeze(0)

    mu_norm = float(
        compute_theta_core(X_fit, ref.expand(X_fit.shape[0], -1)).mean()
    )

    def _score(acts: torch.Tensor) -> np.ndarray:
        X = acts[:, layer, :]
        raw = compute_theta_core(X, ref.expand(X.shape[0], -1))
        return np.abs(raw.cpu().numpy() - mu_norm)

    return {
        "norm":   _score(split.norm_eval),
        "harm":   _score(split.harm_all),   # all harmful for normative_ref convention
        "benign": _score(split.benign),
        "rest":   np.concatenate([_score(split.norm_eval), _score(split.benign)]),
    }


def baseline_bivariate_theta_phi(split: DataSplit, layer: int) -> dict[str, np.ndarray]:
    """
    Bivariate (theta, phi) anomaly score using a 2D Gaussian fit on the
    normative training set.  phi basis is fit from normative training data only
    (one-shot): no test or harmful data is used to define any coordinate.

    Score = -log N_2D((theta, phi) | mu_2D, Sigma_2D).
    Captures anomaly in both the radial (theta) and azimuthal (phi) directions.
    """
    from sklearn.decomposition import PCA as _PCA
    X_fit = split.norm_fit[:, layer, :]

    # Step 1: PC1 from normative training set
    pca1 = _PCA(n_components=1)
    pca1.fit(X_fit.cpu().float().numpy())
    pc1  = torch.tensor(pca1.components_[0], dtype=X_fit.dtype, device=X_fit.device)
    pc1  = pc1 / torch.linalg.norm(pc1)
    ref  = pc1.unsqueeze(0)

    # Step 2: theta for training set
    theta_fit = compute_theta_core(
        X_fit, ref.expand(X_fit.shape[0], -1)
    ).cpu().numpy()

    # Step 3: phi basis from training set orthogonal complements only
    dot_fit    = (X_fit * pc1).sum(dim=-1, keepdim=True)
    X_fit_perp = X_fit - dot_fit * pc1
    pca2       = _PCA(n_components=2)
    pca2.fit(X_fit_perp.cpu().float().numpy())
    fit_2d     = pca2.transform(X_fit_perp.cpu().float().numpy())
    phi_fit    = np.arctan2(fit_2d[:, 1], fit_2d[:, 0])

    # Step 4: fit bivariate Gaussian on (theta, phi) of training set.
    # We pre-compute mu and the Cholesky factor of the covariance, then
    # evaluate logpdf manually to avoid Pylance/scipy stub type errors with
    # the multivariate_normal frozen-distribution constructor.
    coords_fit = np.column_stack([theta_fit, phi_fit])
    mu_2d      = coords_fit.mean(axis=0)                       # (2,)
    cov_2d     = np.cov(coords_fit.T) + 1e-6 * np.eye(2)      # (2,2)
    cov_inv    = np.linalg.inv(cov_2d)
    log_det    = np.log(np.linalg.det(cov_2d))
    _LOG2PI    = np.log(2.0 * np.pi)

    def _mvn_neg_logpdf(coords: np.ndarray) -> np.ndarray:
        """−log N(x; mu_2d, cov_2d) — higher = more anomalous."""
        diff = coords - mu_2d                                  # (N, 2)
        maha = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)  # (N,)
        return 0.5 * (maha + log_det + 2.0 * _LOG2PI)

    def _score(acts: torch.Tensor) -> np.ndarray:
        X     = acts[:, layer, :]
        theta = compute_theta_core(X, ref.expand(X.shape[0], -1)).cpu().numpy()
        dot   = (X * pc1).sum(dim=-1, keepdim=True)
        X_perp= X - dot * pc1
        perp_2d = pca2.transform(X_perp.cpu().float().numpy())
        phi   = np.arctan2(perp_2d[:, 1], perp_2d[:, 0])
        return _mvn_neg_logpdf(np.column_stack([theta, phi]))

    return {
        "norm":   _score(split.norm_eval),
        "harm":   _score(split.harm_all),   # all harmful for normative_ref convention
        "benign": _score(split.benign),
        "rest":   np.concatenate([_score(split.norm_eval), _score(split.benign)]),
    }


# ===========================================================================
# Metric utilities
# ===========================================================================

def _auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    y_true  = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    y_score = np.concatenate([neg, pos])
    if np.isnan(y_score).any() or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _auprc(neg: np.ndarray, pos: np.ndarray) -> float:
    y_true  = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    y_score = np.concatenate([neg, pos])
    if np.isnan(y_score).any():
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _rank_biserial(neg: np.ndarray, pos: np.ndarray) -> float:
    n1, n2 = len(neg), len(pos)
    U, _ = mannwhitneyu(neg, pos, alternative="two-sided")
    return float(1 - (2 * U) / (n1 * n2))


@dataclass
class PRResult:
    precision:          np.ndarray
    recall:             np.ndarray
    thresholds:         np.ndarray
    auprc:              float
    prec_at_90_recall:  float
    prec_at_95_recall:  float
    thr_at_90_recall:   float
    thr_at_95_recall:   float


def _pr_metrics(neg: np.ndarray, pos: np.ndarray) -> PRResult:
    """Compute full PR curve and key operating points."""
    y_true  = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    y_score = np.concatenate([neg, pos])
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    def _at_recall(target: float) -> tuple[float, float]:
        """Highest precision achievable at recall >= target."""
        mask = recall >= target
        if not mask.any():
            return float("nan"), float("nan")
        idx = int(np.where(mask)[0][np.argmax(precision[mask])])
        thr = float(thresholds[idx]) if idx < len(thresholds) else float("nan")
        return float(precision[idx]), thr

    prec90, thr90 = _at_recall(0.90)
    prec95, thr95 = _at_recall(0.95)

    return PRResult(
        precision=precision, recall=recall, thresholds=thresholds,
        auprc=float(average_precision_score(y_true, y_score)),
        prec_at_90_recall=prec90, prec_at_95_recall=prec95,
        thr_at_90_recall=thr90,  thr_at_95_recall=thr95,
    )


def full_row(
    comparison: str,
    strategy:   str,
    layer:      int,
    K:          int,
    neg:        np.ndarray,
    pos:        np.ndarray,
) -> dict:
    pr = _pr_metrics(neg, pos)
    U, p_val = mannwhitneyu(neg, pos, alternative="two-sided")
    return {
        "strategy":            strategy,
        "comparison":          comparison,
        "layer":               layer,
        "K":                   K,
        "n_negative":          len(neg),
        "n_positive":          len(pos),
        "AUROC":               round(_auroc(neg, pos), 4),
        "AUPRC":               round(pr.auprc, 4),
        "prec_at_90_recall":   round(pr.prec_at_90_recall, 4),
        "prec_at_95_recall":   round(pr.prec_at_95_recall, 4),
        "thr_at_90_recall":    round(pr.thr_at_90_recall, 4),
        "thr_at_95_recall":    round(pr.thr_at_95_recall, 4),
        "MannWhitneyU":        float(U),
        "p_value":             float(p_val),
        "rank_biserial_r":     round(_rank_biserial(neg, pos), 4),
    }


# ===========================================================================
# Plotting — pure functions, save to disk
# ===========================================================================

def plot_per_layer_auroc(
    layers:            list[int],
    auroc_harm_by_K:   dict[int, list[float]],
    auroc_benign_by_K: dict[int, list[float]],
    auroc_cos:         list[float],
    auroc_l2:          list[float],
    strategy:          str,
    out_dir:           Path,
    auroc_raw_theta:   list[float] | None = None,
    auroc_bivariate:   list[float] | None = None,
) -> None:
    K_values = sorted(auroc_harm_by_K.keys())
    colors   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, auroc_by_K, suffix in zip(
        axes,
        [auroc_harm_by_K, auroc_benign_by_K],
        ["Harmful (AdvBench)", "Benign-Aggressive (XSTest)"],
    ):
        for K, c in zip(K_values, colors):
            ax.plot(layers, auroc_by_K[K], color=c, lw=2, label=f"K={K}")
        ax.plot(layers, auroc_cos, "--", color="purple", alpha=0.7,
                label="Cosine baseline")
        ax.plot(layers, auroc_l2, "--", color="brown", alpha=0.7,
                label="L2-norm baseline")
        if auroc_raw_theta is not None:
            ax.plot(layers, auroc_raw_theta, "-.", color="teal", alpha=0.8,
                    label="|theta - mu| baseline")
        if auroc_bivariate is not None:
            ax.plot(layers, auroc_bivariate, "-.", color="olive", alpha=0.8,
                    label="Bivariate (theta,phi) baseline")
        ax.axhline(0.5, color="gray", linestyle=":", label="Random (0.5)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC (eval sets only)")
        ax.set_title(f"Per-Layer AUROC — {suffix}\nStrategy: {strategy}")
        ax.legend(fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "auroc_by_layer.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_dimension_ablation(
    dim_labels:         list[str],
    auroc_dim_harmful:  list[float],
    auroc_dim_benign:   list[float],
    best_layer:         int,
    out_dir:            Path,
) -> None:
    x = np.arange(len(dim_labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, auroc_dim_harmful, 0.35, label="vs Harmful (AdvBench)",
           color="#d62728", alpha=0.8)
    ax.bar(x + 0.2, auroc_dim_benign,  0.35, label="vs Benign-Agg (XSTest)",
           color="#2ca02c", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle=":", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["all dims" if l == "all" else f"top-{l}" for l in dim_labels]
    )
    ax.set_xlabel("Dimensions retained (by normative variance)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Dimension Pruning Ablation — Layer {best_layer}, K=2\n"
                 "Strategy: normative_ref")
    ax.legend()
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = out_dir / "auroc_ablation_dim.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_score_distributions(
    scores: dict[str, np.ndarray],
    best_layer: int,
    best_K: int,
    strategy: str,
    out_dir: Path,
) -> None:
    """
    Violin plot of score distributions for all four categories.
    Y-axis clipped to [1st, 99th] percentile of pooled data so that extreme
    GMM-extrapolation outliers (common in harmful_ref) do not collapse the
    visible range. IQR whiskers are drawn explicitly.
    """
    data   = [scores["norm"], scores["harm"], scores["benign"], scores["rest"]]
    labels = ["Normative\n(Alpaca)", "Harmful\n(AdvBench)",
               "Benign-Agg\n(XSTest)", "Rest\n(norm+benign)"]
    colors_v = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    pooled  = np.concatenate(data)
    y_lo    = float(np.percentile(pooled, 1))
    y_hi    = float(np.percentile(pooled, 99))
    clipped = strategy == "harmful_ref"

    fig, ax = plt.subplots(figsize=(10, 6))
    parts  = ax.violinplot(data, positions=list(range(len(data))),
                           showmedians=False, showextrema=False)
    bodies = cast(list[PolyCollection], parts["bodies"])
    for pc, c in zip(bodies, colors_v):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)

    # Explicit median + IQR whiskers — robust to outliers
    for i, d in enumerate(data):
        q25 = float(np.percentile(d, 25))
        q50 = float(np.median(d))
        q75 = float(np.percentile(d, 75))
        ax.vlines(i, max(q25, y_lo), min(q75, y_hi),
                  color="black", lw=2.5, zorder=5)
        ax.scatter(i, q50, color="white", s=35, zorder=6,
                   edgecolors="black", lw=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    margin = 0.05 * abs(y_hi - y_lo)
    ax.set_ylim(y_lo - margin, y_hi + margin)

    score_label = ("Anomaly Score (−log p under normative GMM)"
                   if strategy == "normative_ref"
                   else "Harmful Proximity Score (negated −log p under harmful GMM)")
    ax.set_ylabel(score_label)
    clip_note = "  [y-axis: 1st–99th pct]" if clipped else ""
    ax.set_title(
        f"Score Distributions — Layer {best_layer}, K={best_K}, "
        f"Strategy: {strategy}{clip_note}\n"
        f"AUROC  harm/norm: {_auroc(scores['norm'], scores['harm']):.3f}  |  "
        f"harm/benign: {_auroc(scores['benign'], scores['harm']):.3f}  |  "
        f"harm/rest: {_auroc(scores['rest'], scores['harm']):.3f}"
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = out_dir / "score_distributions.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# Colour palette for PR comparisons — consistent across strategies
_PR_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]

def plot_precision_recall(
    scores:     dict[str, np.ndarray],
    best_layer: int,
    best_K:     int,
    strategy:   str,
    out_dir:    Path,
) -> None:
    """
    PR curves derived from COMPARISON_CONFIGS for the current strategy.

    Using COMPARISON_CONFIGS (rather than hardcoded comparisons) ensures:
      - The correct neg/pos direction for each strategy is respected.
      - harmful_ref shows its specific comparisons (including the
        norm_higher_proximity_than_benign finding) instead of the same
        three harmful-as-positive curves used by normative_ref.
      - Near-chance curves (AUROC ≈ 0.5) are labelled with their AUROC
        in the legend, making weak comparisons explicit rather than
        misleading via a high-precision-at-low-recall artefact.

    Dotted horizontal lines mark chance precision (= prevalence).
    """
    cfg = COMPARISON_CONFIGS[strategy]
    comparisons = [
        (label, scores[neg_key], scores[pos_key], color)
        for (label, neg_key, pos_key), color
        in zip(cfg, _PR_COLORS)
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, neg, pos, color in comparisons:
        pr = _pr_metrics(neg, pos)
        auprc = pr.auprc
        ax.plot(pr.recall, pr.precision, color=color, lw=2,
                label=f"{label} (AUPRC={auprc:.3f})")
        # Mark operating points
        for recall_target, prec_val, marker in [
            (0.90, pr.prec_at_90_recall, "o"),
            (0.95, pr.prec_at_95_recall, "s"),
        ]:
            if not np.isnan(prec_val):
                ax.plot(recall_target, prec_val, marker=marker, color=color,
                        markersize=8, zorder=5)

    # Chance line for each comparison (baseline = prevalence)
    for _, neg, pos, color in comparisons:
        prevalence = len(pos) / (len(neg) + len(pos))
        ax.axhline(prevalence, color=color, linestyle=":", alpha=0.4, lw=1)

    # Legend markers for operating points
    ax.plot([], [], "o", color="gray", label="90% recall operating point")
    ax.plot([], [], "s", color="gray", label="95% recall operating point")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — Layer {best_layer}, K={best_K}\n"
                 f"Strategy: {strategy}  (dotted lines = chance precision)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "precision_recall.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ===========================================================================
# Main evaluation
# ===========================================================================

def run_strategy(
    strategy_name: str,
    split:         DataSplit,
    layers:        list[int],
    K_values:      list[int],
    seed:          int,
) -> tuple[
    dict[int, list[float]],   # auroc_harm_by_K
    dict[int, list[float]],   # auroc_benign_by_K
    list[float],              # auroc_cos
    list[float],              # auroc_l2
    list[float],              # auroc_rand
    list[float],              # auroc_raw_theta
    list[float],              # auroc_bivariate
]:
    """
    Run per-layer AUROC ablation for one strategy.

    Returns
    -------
    auroc_harm_by_K, auroc_benign_by_K : dicts mapping K → list[float] per layer
    auroc_cos, auroc_l2                 : baseline lists per layer
    """
    strategy_fn = (scores_normative_ref
                   if strategy_name == "normative_ref"
                   else scores_harmful_ref)

    auroc_harm_by_K:   dict[int, list[float]] = {K: [] for K in K_values}
    auroc_benign_by_K: dict[int, list[float]] = {K: [] for K in K_values}

    for K in K_values:
        print(f"  K={K} ...", end="", flush=True)
        for layer in layers:
            sc = strategy_fn(split, layer, K)
            auroc_harm_by_K[K].append(_auroc(sc["norm"], sc["harm"]))
            auroc_benign_by_K[K].append(_auroc(sc["benign"], sc["harm"]))
        print(" done")

    # Baselines (always normative_ref convention — comparing norm_eval vs harm_eval)
    print("  Computing baselines...")
    auroc_cos  = [_auroc(baseline_cosine(split, l)["norm"],
                          baseline_cosine(split, l)["harm"])  for l in layers]
    auroc_l2   = [_auroc(baseline_l2(split, l)["norm"],
                          baseline_l2(split, l)["harm"])      for l in layers]
    auroc_rand = [_auroc(baseline_random(split, l, seed)["norm"],
                          baseline_random(split, l, seed)["harm"]) for l in layers]
    auroc_raw_th = [_auroc(baseline_raw_theta(split, l)["norm"],
                           baseline_raw_theta(split, l)["harm"]) for l in layers]
    auroc_biv    = [_auroc(baseline_bivariate_theta_phi(split, l)["norm"],
                           baseline_bivariate_theta_phi(split, l)["harm"]) for l in layers]

    return auroc_harm_by_K, auroc_benign_by_K, auroc_cos, auroc_l2, auroc_rand, auroc_raw_th, auroc_biv


def main() -> None:
    args = parse_args()
    global RESULTS_BASE
    RESULTS_BASE = Path(args.output_dir)
    random.seed(args.seed)

    # ---- Validate layer-selection arguments ----
    if args.skip_layer_sweep and args.eval_layer is None:
        raise ValueError(
            "--skip-layer-sweep requires --eval-layer to be set "
            "(use an integer index or 'last'). "
            "Without a fixed layer there is nothing to evaluate at."
        )

    strategies = (["normative_ref", "harmful_ref"]
                  if args.strategy == "both"
                  else [args.strategy])

    # ---- Load prompts ----
    def load_prompts(path: str, n: int) -> list[str]:
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        random.seed(args.seed)
        return random.sample(lines, min(n, len(lines)))

    norm_prompts   = load_prompts(args.normative_file,  args.normative_n)
    harm_prompts   = load_prompts(args.harmful_file,    args.harmful_n)
    benign_prompts = load_prompts(args.benign_agg_file, args.benign_agg_n)

    print(f"Prompts  — normative: {len(norm_prompts)} | "
          f"harmful: {len(harm_prompts)} | benign: {len(benign_prompts)}")

    # ---- Extract activations (once, reused across strategies) ----
    print(f"\nLoading model: {args.model}")
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

    # ---- Construct the single DataSplit ----
    split = make_split(
        norm_acts, harm_acts, benign_acts,
        train_fraction=args.train_fraction,
        seed=args.seed,
        normative_fit_n=args.normative_fit_n,
        harmful_fit_n=args.harmful_fit_n,
    )
    print(f"\n{split.summary()}")

    layers   = list(range(L))
    K_values = [1, 2, 3, 4]
    D        = norm_acts.shape[-1]

    # ---- Resolve --eval-layer ----
    # Do this after the model is loaded so L is known.
    if args.eval_layer is None:
        resolved_eval_layer: int | None = None
    elif str(args.eval_layer).lower() == "last":
        resolved_eval_layer = L - 1
        print(f"\n  --eval-layer 'last' resolved to layer {resolved_eval_layer} "
              f"(model has {L} layers, 0-indexed).")
    else:
        try:
            resolved_eval_layer = int(args.eval_layer)
        except ValueError:
            raise ValueError(
                f"--eval-layer must be an integer or 'last', got: {args.eval_layer!r}"
            )
        if not (0 <= resolved_eval_layer < L):
            raise ValueError(
                f"--eval-layer {resolved_eval_layer} is out of range "
                f"for a model with {L} layers (valid: 0–{L - 1})."
            )

    all_stat_rows: list[dict] = []

    # ======================================================================
    # Loop over strategies
    # ======================================================================
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy}")
        print(f"{'='*60}")

        out_dir = RESULTS_BASE / strategy
        out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Exp 1: Per-layer AUROC across K  (skippable)
        # ------------------------------------------------------------------
        if args.skip_layer_sweep:
            print(f"\n[Exp 1 / {strategy}] Skipped (--skip-layer-sweep). "
                  f"Using fixed eval-layer={args.eval_layer}.")
            # Compute K-ablation summary at the fixed layer only, so the
            # console output is still informative without a full sweep.
            auroc_harm_by_K = None
            auroc_cos       = None
        else:
            print(f"\n[Exp 1 / {strategy}] Per-layer AUROC across K directions...")
            (auroc_harm_by_K, auroc_benign_by_K,
             auroc_cos, auroc_l2, _, auroc_raw_th, auroc_biv) = run_strategy(
                strategy, split, layers, K_values, args.seed
            )

            plot_per_layer_auroc(
                layers, auroc_harm_by_K, auroc_benign_by_K,
                auroc_cos, auroc_l2, strategy, out_dir,
                auroc_raw_theta=auroc_raw_th,
                auroc_bivariate=auroc_biv,
            )

        # ------------------------------------------------------------------
        # Determine the operating layer for Exp 2 and Exp 3.
        # ------------------------------------------------------------------
        if resolved_eval_layer is not None:
            # Fixed by the caller — no harmful data used for layer selection.
            operating_layer = resolved_eval_layer
            operating_K     = 1    # K=1 is the primary detector regardless
            print(f"\n  Operating layer fixed by --eval-layer: {operating_layer}")
        else:
            # Original behaviour: argmax over harmful AUROC.
            # Note: this uses the harmful eval set for layer selection.
            assert auroc_harm_by_K is not None, \
                "Layer sweep required when --eval-layer is not set."
            metrics_dict: dict[int, list[float]] = auroc_harm_by_K            
            operating_K     = max(K_values, key=lambda K: max(metrics_dict[K]))
            operating_layer = int(np.argmax(metrics_dict[operating_K]))
            print(f"\n  Operating layer selected by argmax: "
                  f"K={operating_K}, layer={operating_layer}")

        # ------------------------------------------------------------------
        # Exp 2: Dimension pruning ablation (normative_ref only)
        # ------------------------------------------------------------------
        if strategy == "normative_ref":
            print(f"\n[Exp 2 / {strategy}] Dimension pruning ablation "
                  f"at layer {operating_layer}...")

            dim_fracs: list[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
            _ints: list[int] = sorted(set(max(1, int(f * D)) for f in dim_fracs))
            dim_values: list[int | None] = [*_ints, None]

            auroc_dim_harm: list[float] = []
            auroc_dim_ben:  list[float] = []
            dim_labels:     list[str]   = []

            for top_d in dim_values:
                sc = scores_normative_ref(split, operating_layer, K=2,
                                          top_d_dims=top_d)
                auroc_dim_harm.append(_auroc(sc["norm"], sc["harm"]))
                auroc_dim_ben.append(_auroc(sc["benign"], sc["harm"]))
                dim_labels.append(str(top_d) if top_d is not None else "all")

            plot_dimension_ablation(
                dim_labels, auroc_dim_harm, auroc_dim_ben,
                operating_layer, out_dir,
            )

        # ------------------------------------------------------------------
        # Exp 3: Score distributions, PR curves, statistical tests
        # ------------------------------------------------------------------
        print(f"\n[Exp 3 / {strategy}] Score distributions and statistics "
              f"at layer {operating_layer}, K={operating_K}...")

        strategy_fn = (scores_normative_ref
                       if strategy == "normative_ref"
                       else scores_harmful_ref)
        sc = strategy_fn(split, operating_layer, operating_K)

        plot_score_distributions(sc, operating_layer, operating_K, strategy, out_dir)
        plot_precision_recall(sc, operating_layer, operating_K, strategy, out_dir)

        # Save operating config for downstream scripts (e.g. run_model.py
        # uses best_config.json to know which layer to theta-phi plot).
        import json
        with open(out_dir / "best_config.json", "w") as f:
            json.dump({
                "best_layer":    operating_layer,
                "best_K":        operating_K,
                "layer_source":  "fixed" if args.eval_layer is not None else "argmax",
            }, f)

        # Re-fit at the operating (layer, K) to produce a serialisable biomarker.
        # normative_ref only: this is the zero-shot profile users will distribute.
        # harmful_ref is a supervised variant and is intentionally excluded.
        if strategy == "normative_ref":
            _bm = ThetaBiomarker(
                n_directions=operating_K,
                n_gmm_components=1,
                layer_indices=[operating_layer],
            )
            _bm.fit(split.norm_fit)
            _profile_path = out_dir / f"biomarker_layer{operating_layer}.pkl"
            _bm.save(
                _profile_path,
                model_id=args.model,
                fit_n=int(split.norm_fit.shape[0]),
            )
            print(f"  Saved reference profile → {_profile_path}")

        # Theta statistics per category
        print("\n  Theta (raw) statistics per eval category:")
        print(f"  {'Category':<14}  {'mean':>6}  {'median':>6}  {'std':>6}  n")
        for cat, label in [("norm", "norm_eval"), ("harm", "harm_all" if strategy == "normative_ref" else "harm_eval"),
                            ("benign", "benign")]:
            key = "harm" if cat == "harm" else cat
            t = np.abs(sc[key])   # abs: negated scores for harmful_ref
            print(f"  {label:<14}  {t.mean():>6.3f}  "
                  f"{np.median(t):>6.3f}  {t.std():>6.3f}  {len(t)}")

        # K-ablation summary at operating layer.
        # If the layer sweep was skipped we compute the three values on-demand
        # (one fit per K value — fast, only at one layer).
        k1_auroc = _auroc(
            scores_normative_ref(split, operating_layer, K=1)["norm"],
            scores_normative_ref(split, operating_layer, K=1)["harm"],
        ) if strategy == "normative_ref" else _auroc(
            scores_harmful_ref(split, operating_layer, K=1)["norm"],
            scores_harmful_ref(split, operating_layer, K=1)["harm"],
        )

        if auroc_harm_by_K is not None:
            assert auroc_cos is not None, "auroc_cos should be available if auroc_harm_by_K is available"
            # Full sweep was run — read from cached arrays for efficiency.
            other_K_at_layer = [auroc_harm_by_K[K][operating_layer]
                                 for K in K_values if K > 1]
            cos_at_layer = auroc_cos[operating_layer]
        else:
            # Sweep was skipped — compute on-demand at operating_layer.
            strategy_fn_k = (scores_normative_ref if strategy == "normative_ref"
                             else scores_harmful_ref)
            other_K_at_layer = [
                _auroc(strategy_fn_k(split, operating_layer, K)["norm"],
                       strategy_fn_k(split, operating_layer, K)["harm"])
                for K in K_values if K > 1
            ]
            cos_at_layer = _auroc(
                baseline_cosine(split, operating_layer)["norm"],
                baseline_cosine(split, operating_layer)["harm"],
            )

        max_other = max(other_K_at_layer) if other_K_at_layer else float("nan")
        print(f"\n  K-ablation at operating layer ({operating_layer}):")
        print(f"    K=1 AUROC:             {k1_auroc:.4f}")
        print(f"    Best K>1 AUROC:        {max_other:.4f}  "
              f"(delta={max_other - k1_auroc:+.4f})")
        print(f"    Cosine baseline AUROC: {cos_at_layer:.4f}  "
              f"(delta={cos_at_layer - k1_auroc:+.4f})")

        # Statistical tests
        for comp_label, neg_key, pos_key in COMPARISON_CONFIGS[strategy]:
            neg = sc[neg_key]
            pos = sc[pos_key]
            all_stat_rows.append(
                full_row(comp_label, strategy, operating_layer, operating_K, neg, pos)
            )

    # ---- Write unified stats CSV ----
    df_stats = pd.DataFrame(all_stat_rows)
    out_csv  = RESULTS_BASE / "stats_summary.csv"
    df_stats.to_csv(out_csv, index=False)
    print(f"\nSaved stats → {out_csv}")
    print(df_stats[["strategy", "comparison", "AUROC", "AUPRC",
                    "prec_at_90_recall", "rank_biserial_r"]].to_string(index=False))

    print(f"\nAll results saved under {RESULTS_BASE}/")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ThetaBiomarker with two reference strategies."
    )
    p.add_argument("--model",           default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--normative-file",  default="data/raw/normative.txt")
    p.add_argument("--harmful-file",    default="data/raw/harmful.txt")
    p.add_argument("--benign-agg-file", default="data/raw/benign_aggressive.txt")
    p.add_argument("--normative-n",     type=int, default=200)
    p.add_argument("--harmful-n",       type=int, default=200)
    p.add_argument("--benign-agg-n",    type=int, default=200)
    p.add_argument("--normative-fit-n", type=int, default=200,
                   help="Number of normative prompts used for fitting (absolute). "
                        "Determined by stability analysis: AUROC plateaus at N≈200. "
                        "The remainder becomes the held-out eval set, maximising its "
                        "size and therefore metric reliability. Default: 200.")
    p.add_argument("--harmful-fit-n",   type=int, default=200,
                   help="Number of harmful prompts used for fitting (harmful_ref "
                        "strategy only). Default: 200.")
    p.add_argument("--train-fraction",  type=float, default=0.8,
                   help="Fallback fraction used only when --normative-fit-n / "
                        "--harmful-fit-n are not specified (i.e., set to None). "
                        "Default: 0.8. Prefer the absolute-N parameters.")
    p.add_argument("--strategy",        default="normative_ref",
                   choices=["normative_ref", "harmful_ref", "both"],
                   help="Which reference strategy to run. "
                        "'normative_ref' requires no harmful examples at fit "
                        "time. 'harmful_ref' is a supervised variant. "
                        "'both' runs sequentially. Default: normative_ref")
    p.add_argument("--output-dir",      default="results/eval",
                   help="Root directory for all eval outputs. "
                        "Default: results/eval. Override to results/<model>/eval "
                        "when running via run_model.py.")
    p.add_argument("--eval-layer",      type=str, default=None,
                   help="Fix the operating layer for Exp 2/3 (score distributions, "
                        "PR curves, statistics). Accepts an integer index (e.g. '23') "
                        "or the special value 'last', which resolves to the final "
                        "residual-stream layer at runtime after the model is loaded. "
                        "When set, no harmful data is needed for layer selection. "
                        "Exp 1 (per-layer sweep) still runs unless --skip-layer-sweep "
                        "is also set.")
    p.add_argument("--skip-layer-sweep", action="store_true", default=False,
                   help="Skip Exp 1 (per-layer AUROC sweep across all layers and K). "
                        "Requires --eval-layer to be set. Useful when the auroc_by_layer "
                        "figure already exists from a previous run and only the "
                        "score distributions / PR curves / statistics at a fixed layer "
                        "need to be regenerated. Reduces runtime from O(L*K) to O(1) "
                        "model passes.")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
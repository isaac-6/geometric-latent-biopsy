"""
theta.py  (v2)
--------------
Unified geometric anomaly detector for LLM residual streams.

Core idea
---------
The normative (safe) distribution of prompt activations lies on a structured
sub-manifold of the residual stream.  Deviations from this manifold are
captured by projecting each activation onto the top-K principal directions of
the normative distribution and computing the angular deviation (Theta) to each.

The resulting K-dimensional angle vector is modelled by a Gaussian Mixture
Model (GMM) fit on the normative set.  At inference time, the anomaly score is
the *negative log-likelihood* under this GMM — a single scalar suitable for
AUROC computation, regardless of K.

When K=1 and the GMM has one component, this reduces exactly to the original
single-Theta, single-centroid method, preserving backward compatibility.

Key design choices
------------------
* Zero-shot in the harmful direction: the normative set contains ONLY safe
  prompts.  No harmful examples are seen at fit time.
* Dimension pruning: optionally restrict to the top-D dimensions of the
  normative set by variance before computing PCA.  This acts as a filter for
  dimensions that carry structured safe-domain signal.
* Circular-aware embedding: phi-like angles are embedded as (sin, cos) before
  GMM fitting to avoid wrap-around artefacts.
* Per-layer or stacked: the biomarker can operate on a single layer or on a
  concatenation of layers (set `layers` at fit time).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Low-level math
# ---------------------------------------------------------------------------

def compute_theta_core(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle (radians) between row vectors in `x` and `reference`.

    Supports broadcasting, e.g. x: (N, D), reference: (M, D) → (N, M) if
    expanded appropriately, or x: (N, D), reference: (1, D) → (N,).

    Args:
        x         : (..., D) float tensor
        reference : (..., D) float tensor, broadcast-compatible with x

    Returns:
        theta     : (...,) float tensor in [0, π]
    """
    dot_prod = (x * reference).sum(dim=-1)
    norm_x   = torch.linalg.norm(x,         dim=-1)
    norm_ref = torch.linalg.norm(reference,  dim=-1)
    denom    = norm_x * norm_ref

    cos_theta = torch.full_like(dot_prod, float("nan"))
    valid     = denom > torch.finfo(x.dtype).eps
    cos_theta[valid] = dot_prod[valid] / denom[valid]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ThetaBiomarker:
    """
    Geometric anomaly detector based on angular deviation in the residual stream.

    Parameters
    ----------
    n_directions : int
        Number of principal directions of the normative set to use as
        reference axes (K in the paper).  K=1 recovers the original method.
    n_gmm_components : int
        Number of components in the GMM fitted on angle features.
    top_d_dims : int | None
        If set, restrict activations to the top-`top_d_dims` dimensions by
        normative variance before PCA.  None means use all dimensions.
    layer_indices : list[int] | None
        Which layers to use.  None means use all layers (the biomarker
        receives a single (layers, dim) tensor and processes each layer
        independently, then concatenates angle vectors).
    random_state : int
        Seed for GMM fitting reproducibility.
    """

    def __init__(
        self,
        n_directions:     int            = 1,
        n_gmm_components: int            = 1,
        top_d_dims:       Optional[int]  = None,
        layer_indices:    Optional[list] = None,
        random_state:     int            = 42,
    ):
        self.n_directions     = n_directions
        self.n_gmm_components = n_gmm_components
        self.top_d_dims       = top_d_dims
        self.layer_indices    = layer_indices
        self.random_state     = random_state

        # Fitted attributes
        self._dim_indices:   Optional[list]  = None   # selected dimension indices per layer
        self._pca_models:    Optional[list]  = None   # one PCA per layer
        self._gmm:           Optional[GaussianMixture] = None
        self._scaler:        Optional[StandardScaler]  = None
        self._n_layers:      Optional[int]   = None
        self.centroids:      Optional[torch.Tensor] = None  # kept for back-compat

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, normative_activations: torch.Tensor):
        """
        Fit the biomarker on normative (safe) activations.

        Args:
            normative_activations : (N, L, D) tensor
                N samples, L layers, D hidden dimensions.
        """
        if not isinstance(normative_activations, torch.Tensor):
            raise TypeError("normative_activations must be a torch.Tensor")
        if normative_activations.ndim != 3:
            raise ValueError(
                f"Expected (N, L, D) tensor, got shape {tuple(normative_activations.shape)}"
            )

        N, L, D = normative_activations.shape
        layers  = self.layer_indices if self.layer_indices is not None else list(range(L))
        self._n_layers = len(layers)

        # Back-compat: store centroid for the first selected layer
        self.centroids = normative_activations[:, layers, :].mean(dim=0)  # (len(layers), D)

        # Guard: reject degenerate data whose centroid has near-zero norm.
        # A centroid norm at or near zero indicates perfectly anti-symmetric
        # activations (e.g. rows that cancel to zero), which is unphysical
        # for real LLM activations and would produce meaningless PC1 directions.
        centroid_norms = torch.linalg.norm(self.centroids, dim=-1)  # (len(layers),)
        if (centroid_norms < 1e-6).any():
            bad = int((centroid_norms < 1e-6).nonzero(as_tuple=True)[0][0])
            raise ValueError(
                f"Normative activations at layer index {layers[bad]} have a "
                f"near-zero norm centroid ({centroid_norms[bad].item():.2e}). "
                "This indicates degenerate (e.g. perfectly anti-symmetric) input. "
                "Provide at least one non-cancelling normative prompt."
            )

        # ---- Per-layer: dimension pruning + PCA ----
        self._dim_indices = []
        self._pca_models  = []
        angle_features_all = []   # will be (N, n_angle_features)

        for layer_idx in layers:
            X = normative_activations[:, layer_idx, :].cpu().float().numpy()  # (N, D)

            # 1. Dimension pruning by normative variance
            if self.top_d_dims is not None and self.top_d_dims < D:
                var    = X.var(axis=0)
                top_di = np.argsort(var)[::-1][: self.top_d_dims]
                X_sel  = X[:, top_di]
            else:
                top_di = np.arange(D)
                X_sel  = X

            self._dim_indices.append(top_di)

            # 2. PCA to extract top-K reference directions
            n_comp = min(self.n_directions, X_sel.shape[1], N - 1)
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            pca.fit(X_sel)
            self._pca_models.append(pca)

            # 3. Compute angles for the normative set at this layer
            angles = self._compute_angles_numpy(X_sel, pca)  # (N, K) — K angles
            angle_features_all.append(angles)

        # Stack across layers: (N, n_layers * K)
        angle_matrix = np.concatenate(angle_features_all, axis=1)  # (N, n_layers * K)

        # 4. Embed angles as (sin, cos) to handle circularity
        angle_embedded = self._embed_angles(angle_matrix)  # (N, 2 * n_layers * K)

        # 5. Scale before GMM
        self._scaler = StandardScaler()
        angle_scaled = self._scaler.fit_transform(angle_embedded)

        # 6. Fit GMM
        n_comp_gmm = min(self.n_gmm_components, N)
        self._gmm = GaussianMixture(
            n_components=n_comp_gmm,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=200,
            n_init=5,
        )
        self._gmm.fit(angle_scaled)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self, activation: torch.Tensor) -> float:
        """
        Compute anomaly score for a single prompt activation.

        Args:
            activation : (L, D) tensor — all layers for one prompt.

        Returns:
            float : negative log-likelihood under the normative GMM.
                    Higher = more anomalous.
        """
        self._check_fitted()
        # _check_fitted() raises if _gmm is None; assert-narrow remaining
        # Optional attributes so Pylance does not flag them as potentially None.
        assert self._scaler is not None
        assert self._gmm is not None
        angles_vec = self._activation_to_angle_vector(activation)  # (1, n_features)
        embedded   = self._embed_angles(angles_vec)
        scaled     = self._scaler.transform(embedded)
        nll        = -self._gmm.score_samples(scaled)[0]
        return float(nll)

    def score_batch(self, activations: torch.Tensor) -> np.ndarray:
        """
        Compute anomaly scores for a batch of prompts.

        Args:
            activations : (N, L, D) tensor.

        Returns:
            scores : (N,) numpy array of negative log-likelihoods.
        """
        self._check_fitted()
        assert self._scaler is not None
        assert self._gmm is not None
        N = activations.shape[0]
        angle_vecs = np.vstack([
            self._activation_to_angle_vector(activations[i]) for i in range(N)
        ])
        embedded = self._embed_angles(angle_vecs)
        scaled   = self._scaler.transform(embedded)
        return -self._gmm.score_samples(scaled)

    # ------------------------------------------------------------------
    # Per-layer raw angles (for visualisation and ablation)
    # ------------------------------------------------------------------

    def compute_theta(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Back-compatible: return Theta to the first reference direction for
        each layer.  Shape: (n_layers,).
        """
        self._check_fitted()
        assert self._dim_indices is not None
        assert self._pca_models is not None
        layers = self.layer_indices if self.layer_indices is not None \
                 else list(range(activation.shape[0]))

        thetas = []
        for i, layer_idx in enumerate(layers):
            x_np  = activation[layer_idx, :].cpu().float().numpy()
            x_sel = x_np[self._dim_indices[i]]
            pc1   = torch.tensor(
                self._pca_models[i].components_[0], dtype=torch.float32
            )
            x_t   = torch.tensor(x_sel, dtype=torch.float32)
            thetas.append(compute_theta_core(x_t.unsqueeze(0), pc1.unsqueeze(0)).item())

        return torch.tensor(thetas)

    def compute_all_angles(self, activation: torch.Tensor) -> np.ndarray:
        """
        Return the full (n_layers, K) angle matrix for one prompt.
        Useful for the theta-phi plane visualisation with K>=2.
        """
        self._check_fitted()
        assert self._dim_indices is not None
        assert self._pca_models is not None
        layers = self.layer_indices if self.layer_indices is not None \
                 else list(range(activation.shape[0]))
        rows = []
        for i, layer_idx in enumerate(layers):
            x_np  = activation[layer_idx, :].cpu().float().numpy()
            x_sel = x_np[self._dim_indices[i]]
            angles = self._compute_angles_numpy(x_sel[None, :], self._pca_models[i])
            rows.append(angles[0])
        return np.stack(rows)  # (n_layers, K)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_angles_numpy(self, X: np.ndarray, pca: PCA) -> np.ndarray:
        """
        Compute angles between each row of X and each PC of pca.

        Args:
            X   : (N, D) float array
            pca : fitted PCA with K components

        Returns:
            angles : (N, K) float array in [0, π]
        """
        components = pca.components_  # (K, D)
        angles = np.zeros((X.shape[0], components.shape[0]))
        for k, pc in enumerate(components):
            dot   = X @ pc                      # (N,)
            norm_x = np.linalg.norm(X, axis=1)  # (N,)
            norm_c = np.linalg.norm(pc)
            denom  = norm_x * norm_c
            valid  = denom > 1e-8
            cos    = np.where(valid, dot / np.where(valid, denom, 1.0), np.nan)
            cos    = np.clip(cos, -1.0, 1.0)
            angles[:, k] = np.arccos(cos)
        return angles

    def _embed_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Embed each angle θ as (sin θ, cos θ) for circular-aware modelling.

        Args:
            angles : (N, F) float array

        Returns:
            embedded : (N, 2*F) float array
        """
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    def _activation_to_angle_vector(self, activation: torch.Tensor) -> np.ndarray:
        """Single activation (L, D) → (1, n_layers * K) angle array."""
        assert self._dim_indices is not None
        assert self._pca_models is not None
        layers = self.layer_indices if self.layer_indices is not None \
                 else list(range(activation.shape[0]))
        parts = []
        for i, layer_idx in enumerate(layers):
            x_np  = activation[layer_idx, :].cpu().float().numpy()
            x_sel = x_np[self._dim_indices[i]]
            angles = self._compute_angles_numpy(x_sel[None, :], self._pca_models[i])
            parts.append(angles)
        return np.concatenate(parts, axis=1)  # (1, n_layers * K)

    def _check_fitted(self):
        if self._gmm is None:
            raise ValueError("Call fit() before score() or compute_theta().")
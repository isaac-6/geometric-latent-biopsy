"""
tests/test_theta.py
-------------------
Unit tests for src/theta.py — ThetaBiomarker and compute_theta_core.

Test categories
---------------
  A. Pure math (compute_theta_core)          — no model loading needed
  B. Input validation (ThetaBiomarker.fit)   — fast, synthetic tensors
  C. Scoring contract                        — shape, monotonicity, consistency
  D. Layer-index subsetting                  — correct layer selection
  E. K>1 directions                          — multi-direction API
  F. Batch / single consistency              — score vs score_batch agree
  G. High-dimensional precision              — float32 rounding at LLM scale

All tests use synthetic random data; no HuggingFace models are loaded.
"""

import math

import numpy as np
import pytest
import torch

from src.theta import ThetaBiomarker, compute_theta_core


# ---------------------------------------------------------------------------
# A. Pure math — compute_theta_core
# ---------------------------------------------------------------------------

class TestComputeThetaCore:
    def test_identical_vectors_give_zero(self):
        x   = torch.tensor([[1.0, 0.0, 0.0]])
        ref = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.isclose(compute_theta_core(x, ref), torch.tensor(0.0), atol=1e-5)

    def test_opposite_vectors_give_pi(self):
        x   = torch.tensor([[-1.0, 0.0, 0.0]])
        ref = torch.tensor([[1.0,  0.0, 0.0]])
        assert torch.isclose(compute_theta_core(x, ref), torch.tensor(math.pi), atol=1e-5)

    def test_orthogonal_vectors_give_half_pi(self):
        x   = torch.tensor([[0.0, 1.0, 0.0]])
        ref = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.isclose(compute_theta_core(x, ref), torch.tensor(math.pi / 2), atol=1e-5)

    def test_zero_vector_gives_nan(self):
        x   = torch.tensor([[0.0, 0.0, 0.0]])
        ref = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.isnan(compute_theta_core(x, ref))

    def test_output_in_zero_to_pi(self):
        """Theta must lie in [0, π] for all non-degenerate inputs."""
        torch.manual_seed(0)
        x   = torch.randn(50, 16)
        ref = torch.randn(50, 16)
        theta = compute_theta_core(x, ref)
        valid = ~torch.isnan(theta)
        assert (theta[valid] >= 0).all()
        assert (theta[valid] <= math.pi + 1e-5).all()

    def test_batch_shape_preserved(self):
        x   = torch.randn(8, 32)
        ref = torch.randn(8, 32)
        out = compute_theta_core(x, ref)
        assert out.shape == (8,)

    def test_scalar_broadcast(self):
        """Single reference broadcast against a batch."""
        x   = torch.randn(10, 32)
        ref = torch.randn(1,  32)
        out = compute_theta_core(x, ref.expand(10, -1))
        assert out.shape == (10,)


# ---------------------------------------------------------------------------
# B. Input validation — ThetaBiomarker.fit()
# ---------------------------------------------------------------------------

class TestFitValidation:
    def test_wrong_type_raises(self):
        bm = ThetaBiomarker()
        with pytest.raises(TypeError):
            bm.fit([[1.0, 2.0]])  # type: ignore[arg-type]

    def test_wrong_ndim_raises(self):
        bm = ThetaBiomarker()
        with pytest.raises(ValueError):
            bm.fit(torch.randn(5, 16))  # 2D, should be 3D

    def test_near_zero_centroid_raises(self):
        """Perfectly anti-symmetric activations cancel to a zero centroid.

        The guard must raise a ValueError whose message includes 'near-zero norm'.
        This catches degenerate inputs that would produce meaningless PC1 directions.
        """
        bm = ThetaBiomarker()
        # Two rows that cancel: mean = [0, 0, 0]
        acts = torch.tensor([
            [[1.0, 2.0, 3.0]],
            [[-1.0, -2.0, -3.0]],
        ])
        with pytest.raises(ValueError, match="near-zero norm"):
            bm.fit(acts)

    def test_score_before_fit_raises(self):
        bm = ThetaBiomarker()
        with pytest.raises(ValueError, match="fit()"):
            bm.score(torch.randn(4, 16))

    def test_score_batch_before_fit_raises(self):
        bm = ThetaBiomarker()
        with pytest.raises(ValueError):
            bm.score_batch(torch.randn(3, 4, 16))

    def test_minimum_viable_fit(self):
        """fit() should succeed with N=3 samples (minimum for stable PCA + GMM)."""
        bm = ThetaBiomarker()
        bm.fit(torch.randn(3, 2, 16))


# ---------------------------------------------------------------------------
# C. Scoring contract
# ---------------------------------------------------------------------------

class TestScoringContract:
    def _fitted_biomarker(self, n=50, layers=4, dim=64, seed=42):
        torch.manual_seed(seed)
        bm = ThetaBiomarker()
        bm.fit(torch.randn(n, layers, dim))
        return bm, layers, dim

    def test_score_returns_float(self):
        bm, L, D = self._fitted_biomarker()
        s = bm.score(torch.randn(L, D))
        assert isinstance(s, float)

    def test_score_batch_returns_1d_array(self):
        bm, L, D = self._fitted_biomarker()
        scores = bm.score_batch(torch.randn(10, L, D))
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (10,)

    def test_score_and_score_batch_agree(self):
        """score() and score_batch() must return the same values."""
        bm, L, D = self._fitted_biomarker()
        torch.manual_seed(7)
        acts = torch.randn(5, L, D)
        single = np.array([bm.score(acts[i]) for i in range(5)])
        batch  = bm.score_batch(acts)
        np.testing.assert_allclose(single, batch, rtol=1e-5)

    def test_anomaly_score_monotone_with_deviation(self):
        """A prompt whose theta is farther from mu_0 must score higher.

        For random Gaussian normative data, the normative theta distribution
        has mu_0 ≈ π/2 (random vectors are approximately orthogonal to any
        fixed direction in high dimensions).  A vector ORTHOGONAL to PC1 also
        has theta ≈ π/2 — it lands right at mu_0 and is not anomalous.

        Instead we use the PC1 direction itself as the anomalous point:
        theta_far ≈ 0, which is far from mu_0 ≈ π/2, giving a high score.
        The 'near' point is a held-out normative sample with theta ≈ mu_0.
        """
        torch.manual_seed(3)
        n, L, D = 60, 1, 32
        norm_acts = torch.randn(n, L, D)
        bm = ThetaBiomarker(layer_indices=[0])
        bm.fit(norm_acts)

        # near: a held-out normative sample — theta close to mu_0 (low score)
        held_out = torch.randn(1, L, D)

        # far: aligned with PC1 → theta ≈ 0, very far from mu_0 ≈ π/2
        assert bm._pca_models is not None  # narrowed after fit()
        pc1 = torch.tensor(bm._pca_models[0].components_[0], dtype=torch.float32)
        far_act = (pc1 * 20.0).unsqueeze(0).unsqueeze(0)  # (1, 1, D), theta ≈ 0

        assert bm.score(far_act[0]) > bm.score(held_out[0])

    def test_scores_finite(self):
        bm, L, D = self._fitted_biomarker()
        scores = bm.score_batch(torch.randn(20, L, D))
        assert np.isfinite(scores).all()

    def test_compute_theta_shape(self):
        bm, L, D = self._fitted_biomarker()
        theta = bm.compute_theta(torch.randn(L, D))
        assert theta.shape == (L,)

    def test_compute_theta_in_range(self):
        bm, L, D = self._fitted_biomarker()
        theta = bm.compute_theta(torch.randn(L, D))
        assert (theta >= 0).all()
        assert (theta <= math.pi + 1e-4).all()


# ---------------------------------------------------------------------------
# D. Layer-index subsetting
# ---------------------------------------------------------------------------

class TestLayerIndexSubset:
    def test_single_layer_index(self):
        torch.manual_seed(9)
        bm = ThetaBiomarker(layer_indices=[2])
        bm.fit(torch.randn(20, 5, 32))
        scores = bm.score_batch(torch.randn(8, 5, 32))
        assert scores.shape == (8,)

    def test_subset_of_layers(self):
        torch.manual_seed(11)
        bm = ThetaBiomarker(layer_indices=[0, 3])
        bm.fit(torch.randn(20, 5, 32))
        scores = bm.score_batch(torch.randn(6, 5, 32))
        assert scores.shape == (6,)

    def test_layer_indices_vs_all_layers_differ(self):
        """Selecting a subset of layers changes (does not crash) the scores."""
        torch.manual_seed(13)
        acts = torch.randn(30, 6, 32)
        bm_all = ThetaBiomarker()
        bm_sub = ThetaBiomarker(layer_indices=[0, 2, 4])
        bm_all.fit(acts)
        bm_sub.fit(acts)
        probe = torch.randn(6, 32)
        # Different feature sets → different scores (probabilistic; very unlikely equal)
        assert bm_all.score(probe) != bm_sub.score(probe)


# ---------------------------------------------------------------------------
# E. K>1 directions
# ---------------------------------------------------------------------------

class TestKDirections:
    def test_k2_fits_without_error(self):
        bm = ThetaBiomarker(n_directions=2)
        bm.fit(torch.randn(30, 3, 64))
        assert bm.score_batch(torch.randn(5, 3, 64)).shape == (5,)

    def test_k_capped_at_n_minus_1(self):
        """n_directions is silently capped at min(K, D, N-1); no crash."""
        bm = ThetaBiomarker(n_directions=100)
        bm.fit(torch.randn(10, 2, 16))   # N-1=9, D=16 → capped at 9

    def test_compute_all_angles_shape(self):
        torch.manual_seed(5)
        K = 3
        bm = ThetaBiomarker(n_directions=K)
        bm.fit(torch.randn(40, 4, 32))
        angles = bm.compute_all_angles(torch.randn(4, 32))
        assert angles.shape == (4, K)


# ---------------------------------------------------------------------------
# F. Dimension pruning
# ---------------------------------------------------------------------------

class TestDimensionPruning:
    def test_top_d_dims_respected(self):
        bm = ThetaBiomarker(top_d_dims=8)
        bm.fit(torch.randn(30, 2, 64))
        assert bm.score_batch(torch.randn(4, 2, 64)).shape == (4,)

    def test_top_d_dims_larger_than_d_falls_back(self):
        """top_d_dims > D is silently capped; no crash."""
        bm = ThetaBiomarker(top_d_dims=10_000)
        bm.fit(torch.randn(20, 2, 32))


# ---------------------------------------------------------------------------
# G. High-dimensional float32 precision
# ---------------------------------------------------------------------------

class TestHighDimensionalPrecision:
    def test_self_angle_near_zero_at_lm_scale(self):
        """
        compute_theta_core(v, v) must be ≈ 0 for LLM-scale vectors.

        Float32 cosine computation in D=896 introduces ~0.0005 rounding error
        from the dot-product accumulation; we allow atol=1e-3.

        Note: this tests compute_theta_core directly, NOT ThetaBiomarker.
        The centroid (mean activation) and PC1 (direction of max variance) are
        unrelated — the angle between them is not guaranteed to be small.
        Passing the centroid to compute_theta would give theta ≈ π/2 for random
        Gaussian data, since the centroid is near-zero and approximately
        orthogonal to any fixed direction in high dimensions.
        """
        torch.manual_seed(0)
        D = 896
        N = 20
        # N random LLM-scale vectors as both x and reference (self-angle = 0)
        vecs = torch.randn(N, D) * 50.0   # large norm, typical of real activations
        theta = compute_theta_core(vecs, vecs)
        assert torch.allclose(theta, torch.zeros(N), atol=1e-3), \
            f"Max self-angle at D={D}: {theta.max().item():.6f} rad"

    def test_known_angle_preserved_at_lm_scale(self):
        """
        A 90° angle between two orthogonal D=896 vectors must be accurately
        recovered despite float32 accumulation errors.
        """
        torch.manual_seed(1)
        D = 896
        v = torch.randn(D)
        v = v / v.norm()
        # construct a vector orthogonal to v in D dimensions
        w = torch.randn(D)
        w = w - (w @ v) * v
        w = w / w.norm()
        theta = compute_theta_core(v.unsqueeze(0), w.unsqueeze(0))
        assert torch.isclose(theta, torch.tensor(math.pi / 2), atol=1e-3), \
            f"Expected π/2, got {theta.item():.6f}"
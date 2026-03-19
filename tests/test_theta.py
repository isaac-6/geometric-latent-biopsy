import torch
import math
import pytest
from src.theta import compute_theta_core, ThetaBiomarker

def test_trivial_math():
    """Test pure theta math without any LLM context."""
    # Define simple 3D vectors to test the core logic
    x = torch.tensor([[1.0, 0.0, 0.0],   # Will test identical
        [-1.0, 0.0, 0.0],  # Will test opposite
        [0.0, 1.0, 0.0],   # Will test orthogonal
        [0.0, 0.0, 0.0],   # Will test zero norm (should be NaN)
    ])
    
    # Reference vector
    ref = torch.tensor([[1.0, 0.0, 0.0],[1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],[1.0, 0.0, 0.0],
    ])
    
    theta = compute_theta_core(x, ref)
    
    # 1. Identical -> 0.0
    assert torch.isclose(theta[0], torch.tensor(0.0), atol=1e-5), f"Expected 0.0, got {theta[0]}"
    
    # 2. Opposite -> pi
    assert torch.isclose(theta[1], torch.tensor(math.pi), atol=1e-5), f"Expected {math.pi}, got {theta[1]}"
    
    # 3. Orthogonal -> pi/2
    assert torch.isclose(theta[2], torch.tensor(math.pi / 2), atol=1e-5), f"Expected {math.pi/2}, got {theta[2]}"
    
    # 4. Zero vector -> NaN
    assert torch.isnan(theta[3]), f"Expected NaN for zero vector, got {theta[3]}"

def test_biomarker_fit_validation():
    """Ensure the fit process rejects invalid centroids (like all zeros)."""
    biomarker = ThetaBiomarker()
    
    # Create fake activations that perfectly cancel out to zero
    # Batch size 2, 1 layer, 3 dimensions
    zero_activations = torch.tensor([
        [[1.0, 2.0, 3.0]],
        [[-1.0, -2.0, -3.0]]
    ])
    
    # This should raise our safeguard ValueError
    with pytest.raises(ValueError, match="near-zero norm"):
        biomarker.fit(zero_activations)

def test_biomarker_high_dimensional_precision():
    """Test standard float32 precision limits in LLM-sized latents."""
    batch_size = 100
    num_layers = 24
    hidden_dim = 896
    
    # 1. Create dummy normative data
    normative_data = torch.randn(batch_size, num_layers, hidden_dim)
    
    # 2. Fit
    biomarker = ThetaBiomarker()
    biomarker.fit(normative_data)
    
    # 3. Identical activation
    identical_activation = biomarker.centroids.clone()
    theta_identical = biomarker.compute_theta(identical_activation)
    
    # Note the atol=1e-3 here. This handles the 0.0005 float32 rounding error 
    # expected in 896-dimensional space without failing the build.
    assert torch.allclose(theta_identical, torch.zeros(num_layers), atol=1e-3)
    
    print("\n✅ Pure math tests, zero-vector handling, and high-dim precision tests passed!")

if __name__ == "__main__":
    test_trivial_math()
    test_biomarker_fit_validation()
    test_biomarker_high_dimensional_precision()
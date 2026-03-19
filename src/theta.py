import torch

def compute_theta_core(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Core math engine for Theta. Calculates the angle (theta) in radians 
    between row vectors in x and reference. Matches original NumPy implementation.
    """
    # Compute dot product along the last dimension
    dot_prod = (x * reference).sum(dim=-1)
    
    # Compute L2 norms
    norm_x = torch.linalg.norm(x, dim=-1)
    norm_ref = torch.linalg.norm(reference, dim=-1)
    
    # Initialize with NaNs
    cos_theta = torch.full_like(dot_prod, float('nan'))
    
    # Avoid division by zero
    valid_denominator = (norm_x * norm_ref) > torch.finfo(x.dtype).eps
    
    # Calculate valid cosines
    cos_theta[valid_denominator] = dot_prod[valid_denominator] / (norm_x[valid_denominator] * norm_ref[valid_denominator])
    
    # Clip to handle float inaccuracies
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    
    # Calculate theta
    theta = torch.acos(cos_theta)
    
    return theta


class ThetaBiomarker:
    def __init__(self):
        self.centroids = None

    def fit(self, normative_activations: torch.Tensor):
        if not isinstance(normative_activations, torch.Tensor):
            raise TypeError("normative_activations must be a PyTorch Tensor")
            
        if normative_activations.ndim != 3:
            raise ValueError(f"Expected 3D tensor (batch, layers, dim), got {normative_activations.ndim}D")
            
        # Compute the normative centroid
        self.centroids = normative_activations.mean(dim=0)
        
        # Safeguard: Ensure no centroid is a zero vector
        norms = torch.linalg.norm(self.centroids, dim=-1)
        if torch.any(norms < 1e-6):
            raise ValueError("Fit Error: One or more computed centroids have near-zero norm.")

    def compute_theta(self, activation: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise ValueError("The biomarker must be fitted with normative data first. Call fit().")
            
        if activation.ndim != 2:
            raise ValueError(f"Expected 2D tensor (layers, dim), got {activation.ndim}D")
            
        return compute_theta_core(activation, self.centroids)
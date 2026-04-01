"""
LatentBiopsy — geometric anomaly detection in LLM residual streams.

Public API
----------
    from latentbiopsy.extraction import LatentExtractor
    from latentbiopsy.theta import ThetaBiomarker, compute_theta_core
"""

from .extraction import LatentExtractor
from .theta import ThetaBiomarker, compute_theta_core

__all__ = ["LatentExtractor", "ThetaBiomarker", "compute_theta_core"]
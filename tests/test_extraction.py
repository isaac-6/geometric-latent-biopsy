"""
tests/test_extraction.py
------------------------
Integration tests for src/extraction.py — LatentExtractor.

These tests load a real HuggingFace model (Qwen2.5-0.5B-Instruct) and require
network access on first run.  They are marked with the `slow` pytest marker so
they can be skipped in fast CI runs:

    pytest -m "not slow"      # skip model-loading tests
    pytest -m slow            # run only model-loading tests

The model is loaded once per module via a session-scoped fixture to avoid
redundant downloads.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Session-scoped fixture — load the model once for the whole test module
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def extractor():
    from latentbiopsy.extraction import LatentExtractor
    return LatentExtractor(model_name=MODEL_NAME)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLatentExtractor:
    def test_device_is_valid(self, extractor):
        """Device must be 'cpu' or a valid 'cuda' string."""
        assert extractor.device in ("cpu",) or extractor.device.startswith("cuda")

    def test_device_matches_availability(self, extractor):
        """Device must match torch.cuda.is_available()."""
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        # device_map may expand to 'cuda:0' — check prefix
        assert extractor.device == expected or extractor.device.startswith(expected)

    def test_output_shape(self, extractor):
        """Activation tensor must be (num_layers, hidden_dim)."""
        prompt = "Translate 'hello' to Spanish."
        acts = extractor.get_last_token_activations(prompt)
        num_layers = extractor.model.config.num_hidden_layers
        hidden_dim = extractor.model.config.hidden_size
        assert acts.shape == (num_layers, hidden_dim), (
            f"Expected ({num_layers}, {hidden_dim}), got {acts.shape}"
        )

    def test_output_is_tensor(self, extractor):
        acts = extractor.get_last_token_activations("Hello.")
        assert isinstance(acts, torch.Tensor)

    def test_no_nan_values(self, extractor):
        acts = extractor.get_last_token_activations("What is 2 + 2?")
        assert not torch.isnan(acts).any(), "NaN values found in activations"

    def test_no_inf_values(self, extractor):
        acts = extractor.get_last_token_activations("What is 2 + 2?")
        assert not torch.isinf(acts).any(), "Inf values found in activations"

    def test_num_layers_matches_config(self, extractor):
        acts = extractor.get_last_token_activations("Test.")
        assert acts.shape[0] == extractor.num_layers
        assert extractor.num_layers == extractor.model.config.num_hidden_layers

    def test_deterministic_output(self, extractor):
        """Same prompt must produce identical activations on repeated calls."""
        prompt = "The capital of France is"
        acts1 = extractor.get_last_token_activations(prompt)
        acts2 = extractor.get_last_token_activations(prompt)
        assert torch.allclose(acts1, acts2), (
            "Repeated calls with same prompt produced different activations"
        )

    def test_different_prompts_differ(self, extractor):
        """Different prompts must produce different activations."""
        acts_a = extractor.get_last_token_activations("What is the capital of France?")
        acts_b = extractor.get_last_token_activations("Write a Python function to sort a list.")
        assert not torch.allclose(acts_a, acts_b), (
            "Different prompts produced identical activations"
        )

    def test_very_short_prompt(self, extractor):
        """Single-token prompt must not crash."""
        acts = extractor.get_last_token_activations("Hi")
        assert acts.ndim == 2

    def test_longer_prompt(self, extractor):
        """A longer prompt (>50 tokens) must still extract from the last token only."""
        long_prompt = " ".join(["word"] * 60)
        acts = extractor.get_last_token_activations(long_prompt)
        assert acts.shape[0] == extractor.num_layers

    def test_activations_are_not_all_zero(self, extractor):
        """Sanity check: activations should not be the zero vector."""
        acts = extractor.get_last_token_activations("Explain neural networks.")
        assert acts.abs().max() > 1e-4
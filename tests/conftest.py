"""
conftest.py
-----------
Pytest configuration for the LatentBiopsy test suite.

Markers
-------
  slow   : tests that require downloading and loading a HuggingFace model
           (~1 GB on first run, ~10-30 s thereafter).

           Skip these in fast CI / pre-commit:
               pytest -m "not slow"

           Run only these (e.g. nightly):
               pytest -m slow
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests that load a HuggingFace model (deselect with -m 'not slow')",
    )
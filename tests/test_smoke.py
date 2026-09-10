"""Wiring check: pytest runs, and the four packages import.

Nothing here asserts anything about speculative decoding yet -- that is the
point. If this fails, the problem is the environment or the layout, not the
research code, which makes it the right first thing to run in a new session.
"""
import importlib

import pytest

PACKAGES = ["harness", "training", "eval", "analysis"]


def test_pytest_is_wired():
    assert True


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    assert importlib.import_module(name) is not None

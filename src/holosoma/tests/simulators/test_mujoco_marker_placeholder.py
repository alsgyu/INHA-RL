"""Placeholder so the ``mujoco`` CI lane collects at least one test.

The positive backend-marker taxonomy runs each CI lane as ``pytest -m "<backend>"``;
an empty selection makes pytest exit 5 (no tests collected) and fails the lane. The
real MuJoCo tests live under ``tests/simulators/mujoco/`` and arrive in a later branch
of this stack. This file keeps the ``mujoco`` lane green until then and is removed
once those tests exist.
"""

import pytest

pytestmark = pytest.mark.mujoco


def test_mujoco_backend_importable():
    """MuJoCo is importable in the mujoco CI image."""
    import mujoco

    assert mujoco.__version__

"""Configuration types for randomization manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass

from holosoma.config_types.distribution import DistributionLike


@dataclass(frozen=True)
class BaseComRange:
    """Per-axis CoM-offset ranges for ``randomize_base_com_startup`` (added to the torso CoM, metres).

    Each axis is a ``[lo, hi]`` pair (uniform), spec dict, or :class:`DistributionSpec`. Offsets
    are typically signed and small, e.g. ``BaseComRange(x=[-0.025, 0.025], y=[-0.05, 0.05], z=[-0.05, 0.05])``.
    """

    x: DistributionLike = (0.0, 0.0)
    y: DistributionLike = (0.0, 0.0)
    z: DistributionLike = (0.0, 0.0)


@dataclass(frozen=True)
class RandomizationTermCfg:
    """Configuration for a single randomization hook."""

    func: str
    """Import path of the randomization hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class RandomizationManagerCfg:
    """Configuration for the randomization manager."""

    setup_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked every simulation step."""

    ignore_unsupported: bool = False
    """Flag to ignore errors when randomizers are not implemented by a simulator."""

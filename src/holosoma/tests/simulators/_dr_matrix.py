"""Shared DR test helper: a bound TermSampler for calling DR terms directly in tests.

On the full feature branch this module also holds the cross-backend DR *matrix* runners
(``run_robot_dr`` / ``run_object_dr`` / ``run_distribution_dr``) and the ``ModelReader`` protocol;
the object-DR follow-up branch adds those. This scene-objects branch keeps only ``_sampler``, which
the reset-time pose-jitter test needs.
"""

from __future__ import annotations

from holosoma.utils.sampler import STAGE_RESET, TermSampler


def _sampler(env, term: str = "dr_matrix_test") -> TermSampler:
    """A bound TermSampler for calling DR terms directly in tests.

    These checks assert physical effects/bands, not seed reproducibility; TermSampler requires a seed
    (no global-RNG fallback), so fall back to a fixed test seed when the env has none.
    """
    base_seed = getattr(env, "dr_base_seed", None) or 0
    episode = getattr(env, "dr_episode_count", None)
    return TermSampler.bind(base_seed, term, STAGE_RESET, episode)

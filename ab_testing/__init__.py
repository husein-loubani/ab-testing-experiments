"""A/B Testing: source package for the Fast Food and Cookie Cats experiments."""

from ab_testing.config import (
    ALPHA,
    COOKIE_CATS,
    FAST_FOOD,
    FIGURES_CC_DIR,
    FIGURES_DIR,
    FIGURES_FF_DIR,
    RANDOM_SEED,
)

# These are re-exported on purpose so callers can do `from ab_testing import ALPHA`.
__all__ = [
    "ALPHA",
    "COOKIE_CATS",
    "FAST_FOOD",
    "FIGURES_CC_DIR",
    "FIGURES_DIR",
    "FIGURES_FF_DIR",
    "RANDOM_SEED",
]

"""
config.py
Global constants, dataset schemas, and path helpers for the A/B Testing project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT: Path = _project_root()

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
FIGURES_FF_DIR: Path = FIGURES_DIR / "fast_food"
FIGURES_CC_DIR: Path = FIGURES_DIR / "cookie_cats"

RANDOM_SEED: int = 42
ALPHA: float = 0.05


@dataclass(frozen=True)
class FastFoodConfig:
    raw_filename: str = "WA_Marketing-Campaign.csv"
    processed_filename: str = "fast_food_clean.csv"

    col_location_id: str = "LocationID"
    col_week: str = "week"
    col_promotion: str = "Promotion"
    col_sales: str = "SalesInThousands"

    promotions_all: tuple[int, ...] = (1, 2, 3)
    primary_metric: str = "avg_sales"


@dataclass(frozen=True)
class CookieCatsConfig:
    raw_filename: str = "cookie_cats.csv"
    processed_filename: str = "cookie_cats_clean.csv"

    col_userid: str = "userid"
    col_version: str = "version"
    col_sum_gamerounds: str = "sum_gamerounds"
    col_retention_1: str = "retention_1"
    col_retention_7: str = "retention_7"


FAST_FOOD = FastFoodConfig()
COOKIE_CATS = CookieCatsConfig()


def raw_path(filename: str) -> Path:
    return DATA_RAW_DIR / filename


def processed_path(filename: str) -> Path:
    return DATA_PROCESSED_DIR / filename


def figure_path(filename: str) -> Path:
    return FIGURES_DIR / filename

"""
cleaning.py
Data cleaning and preprocessing for Fast Food and Cookie Cats datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ab_testing.config import COOKIE_CATS, FAST_FOOD


@dataclass(frozen=True)
class CleaningReport:
    n_rows_in: int
    n_rows_out: int
    n_rows_dropped: int
    n_duplicates: int
    n_missing: int
    notes: list[str]


def clean_fast_food(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    required = {
        FAST_FOOD.col_location_id,
        FAST_FOOD.col_week,
        FAST_FOOD.col_promotion,
        FAST_FOOD.col_sales,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fast food dataset missing columns: {sorted(missing)}")

    df0 = df.copy()
    before = len(df0)

    n_missing = int(df0[list(required)].isna().any(axis=1).sum())
    n_duplicates = int(df0.duplicated().sum())

    df0[FAST_FOOD.col_week] = pd.to_numeric(df0[FAST_FOOD.col_week], errors="coerce").astype("Int64")
    df0[FAST_FOOD.col_promotion] = pd.to_numeric(df0[FAST_FOOD.col_promotion], errors="coerce").astype("Int64")
    df0[FAST_FOOD.col_sales] = pd.to_numeric(df0[FAST_FOOD.col_sales], errors="coerce")

    df0 = df0[df0[FAST_FOOD.col_promotion].isin(FAST_FOOD.promotions_all)].copy()
    df0 = df0.dropna(subset=list(required)).copy()
    df0 = df0[df0[FAST_FOOD.col_sales] >= 0].copy()

    df0[FAST_FOOD.col_week] = df0[FAST_FOOD.col_week].astype(int)
    df0[FAST_FOOD.col_promotion] = df0[FAST_FOOD.col_promotion].astype(int)

    after = len(df0)

    report = CleaningReport(
        n_rows_in=before,
        n_rows_out=after,
        n_rows_dropped=before - after,
        n_duplicates=n_duplicates,
        n_missing=n_missing,
        notes=[
            f"Missing values in key columns: {n_missing}",
            f"Exact duplicate rows: {n_duplicates}",
            "Filtered Promotion to 1, 2, 3",
            "Coerced numeric columns and removed negative sales",
        ],
    )
    return df0, report


def fast_food_store_level(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly rows to store-level metrics (one row per store)."""
    return (
        df_clean.groupby([FAST_FOOD.col_location_id, FAST_FOOD.col_promotion], as_index=False)
        .agg(
            avg_sales=(FAST_FOOD.col_sales, "mean"),
            n_weeks=(FAST_FOOD.col_week, "nunique"),
            total_sales=(FAST_FOOD.col_sales, "sum"),
        )
    )


def clean_cookie_cats(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    required = {
        COOKIE_CATS.col_userid,
        COOKIE_CATS.col_version,
        COOKIE_CATS.col_sum_gamerounds,
        COOKIE_CATS.col_retention_1,
        COOKIE_CATS.col_retention_7,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cookie Cats dataset missing columns: {sorted(missing)}")

    df0 = df.copy()
    before = len(df0)

    n_missing = int(df0[list(required)].isna().any(axis=1).sum())
    n_duplicates = int(df0.duplicated().sum())
    n_dup_users = int(df0[COOKIE_CATS.col_userid].duplicated().sum())

    df0[COOKIE_CATS.col_userid] = pd.to_numeric(df0[COOKIE_CATS.col_userid], errors="coerce").astype("Int64")
    df0[COOKIE_CATS.col_sum_gamerounds] = pd.to_numeric(df0[COOKIE_CATS.col_sum_gamerounds], errors="coerce")
    df0[COOKIE_CATS.col_version] = df0[COOKIE_CATS.col_version].astype(str).str.strip()

    for col in [COOKIE_CATS.col_retention_1, COOKIE_CATS.col_retention_7]:
        if df0[col].dtype == bool:
            df0[col] = df0[col].astype(int)
        else:
            df0[col] = (
                df0[col].astype(str).str.strip().str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
            )

    df0 = df0.dropna(subset=list(required)).copy()
    df0 = df0[df0[COOKIE_CATS.col_sum_gamerounds] >= 0].copy()

    # Remove duplicate userids (keep first occurrence)
    df0 = df0.drop_duplicates(subset=[COOKIE_CATS.col_userid], keep="first")

    df0[COOKIE_CATS.col_userid] = df0[COOKIE_CATS.col_userid].astype(int)
    df0[COOKIE_CATS.col_retention_1] = df0[COOKIE_CATS.col_retention_1].astype(int)
    df0[COOKIE_CATS.col_retention_7] = df0[COOKIE_CATS.col_retention_7].astype(int)

    after = len(df0)

    report = CleaningReport(
        n_rows_in=before,
        n_rows_out=after,
        n_rows_dropped=before - after,
        n_duplicates=n_duplicates,
        n_missing=n_missing,
        notes=[
            f"Missing values in key columns: {n_missing}",
            f"Exact duplicate rows: {n_duplicates}",
            f"Duplicate userids: {n_dup_users}",
            "Coerced userid and sum_gamerounds to numeric",
            "Normalized retention columns to 0 or 1",
            "Removed negative gamerounds",
            "Deduplicated on userid (keep first)",
        ],
    )
    return df0, report

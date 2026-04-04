"""
stats.py
Statistical testing utilities for A/B experiments.
Welch t-tests, proportion tests, bootstrap CIs, global tests, SRM checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway


def _as_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr)]


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohen_d_unpaired(x: np.ndarray, y: np.ndarray) -> float:
    x = _as_float_array(x)
    y = _as_float_array(y)
    denom = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0)
    if denom == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / denom)


# ---------------------------------------------------------------------------
# Two-sample mean tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ABMeanTestResult:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff_a_minus_b: float
    pct_lift_vs_b: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    cohen_d: float
    method: str = "Welch t-test"


def welch_ci_mean_diff(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    nx, ny = x.size, y.size
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    diff = float(x.mean() - y.mean())
    se = np.sqrt(vx / nx + vy / ny)

    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx * vx) / (nx * nx * (nx - 1)) + (vy * vy) / (ny * ny * (ny - 1))
    df = df_num / df_den if df_den > 0 else min(nx, ny) - 1

    tcrit = stats.t.ppf(1 - alpha / 2, df=df)
    return float(diff - tcrit * se), float(diff + tcrit * se)


def welch_ttest_mean(
    x: np.ndarray, y: np.ndarray,
    group_a: str, group_b: str,
    alpha: float = 0.05,
) -> ABMeanTestResult:
    x = _as_float_array(x)
    y = _as_float_array(y)
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    ci_low, ci_high = welch_ci_mean_diff(x, y, alpha=alpha)

    mean_a, mean_b = float(x.mean()), float(y.mean())
    diff = mean_a - mean_b
    pct_lift = float(diff / mean_b * 100.0) if mean_b != 0 else float("nan")

    return ABMeanTestResult(
        group_a=group_a,
        group_b=group_b,
        n_a=int(x.size),
        n_b=int(y.size),
        mean_a=mean_a,
        mean_b=mean_b,
        diff_a_minus_b=float(diff),
        pct_lift_vs_b=pct_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        t_stat=float(t_stat),
        p_value=float(p_value),
        cohen_d=cohen_d_unpaired(x, y),
    )


# ---------------------------------------------------------------------------
# Two-proportion test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoProportionTestResult:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    rate_a: float
    rate_b: float
    diff_a_minus_b: float
    pct_lift_vs_b: float
    ci_low: float
    ci_high: float
    z_stat: float
    p_value: float
    method: str = "Two-proportion z-test"


def two_proportion_test(
    *,
    success_a: int,
    n_a: int,
    success_b: int,
    n_b: int,
    group_a: str,
    group_b: str,
    alpha: float = 0.05,
) -> TwoProportionTestResult:
    p1 = success_a / n_a
    p2 = success_b / n_b
    diff = p1 - p2

    p_pool = (success_a + success_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else float("nan")
    p_value = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else float("nan")

    se_unpooled = np.sqrt(p1 * (1 - p1) / n_a + p2 * (1 - p2) / n_b)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - zcrit * se_unpooled
    ci_high = diff + zcrit * se_unpooled

    pct_lift = (diff / p2 * 100.0) if p2 != 0 else float("nan")

    return TwoProportionTestResult(
        group_a=group_a,
        group_b=group_b,
        n_a=n_a,
        n_b=n_b,
        rate_a=float(p1),
        rate_b=float(p2),
        diff_a_minus_b=float(diff),
        pct_lift_vs_b=float(pct_lift),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        z_stat=float(z),
        p_value=float(p_value),
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci_mean_diff(
    x: np.ndarray, y: np.ndarray,
    *, alpha: float = 0.05, n_boot: int = 10000, seed: int = 42,
) -> tuple[float, float]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    rng = np.random.default_rng(seed)
    nx, ny = x.size, y.size
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = rng.choice(x, size=nx, replace=True)
        yb = rng.choice(y, size=ny, replace=True)
        diffs[i] = xb.mean() - yb.mean()
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Global and post-hoc tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlobalTestResult:
    method: str
    p_value: float
    stat: float
    details: dict[str, Any]


def choose_global_test(groups: dict[str, np.ndarray], *, alpha: float = 0.05) -> GlobalTestResult:
    arrays = [_as_float_array(v) for v in groups.values()]
    group_names = list(groups.keys())

    try:
        _, p_levene = stats.levene(*arrays, center="median")
    except Exception:
        p_levene = float("nan")

    try:
        max_abs_skew = float(np.nanmax([abs(stats.skew(a)) for a in arrays]))
        max_kurt = float(np.nanmax([stats.kurtosis(a, fisher=True) for a in arrays]))
    except Exception:
        max_abs_skew, max_kurt = float("inf"), float("inf")

    normalish = bool((max_abs_skew <= 1.0) and (max_kurt <= 2.0))
    equal_var = bool((not np.isnan(p_levene)) and (p_levene > alpha))

    if normalish and equal_var:
        stat, p_value = stats.f_oneway(*arrays)
        return GlobalTestResult("One-way ANOVA", float(p_value), float(stat),
                                {"levene_p": float(p_levene), "groups": group_names})

    if normalish and (not equal_var):
        res = anova_oneway(arrays, use_var="unequal", welch_correction=True)
        return GlobalTestResult("Welch ANOVA", float(res.pvalue), float(res.statistic),
                                {"levene_p": float(p_levene), "groups": group_names})

    stat, p_value = stats.kruskal(*arrays)
    return GlobalTestResult("Kruskal-Wallis", float(p_value), float(stat),
                            {"levene_p": float(p_levene), "groups": group_names})


def adjust_pvalues(p_values: np.ndarray, method: Literal["holm", "fdr_bh"] = "holm") -> np.ndarray:
    _, p_adj, _, _ = multipletests(np.asarray(p_values, dtype=float), method=method)
    return p_adj


def pairwise_tests(
    groups: dict[str, np.ndarray],
    *, alpha: float = 0.05, p_adjust: Literal["holm", "fdr_bh"] = "holm",
) -> pd.DataFrame:
    names = list(groups.keys())
    arrays = {k: _as_float_array(v) for k, v in groups.items()}

    rows: list[dict[str, Any]] = []
    pvals: list[float] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = welch_ttest_mean(arrays[a], arrays[b], group_a=a, group_b=b, alpha=alpha)
            rows.append({
                "comparison": f"{a} vs {b}",
                "n_a": res.n_a, "n_b": res.n_b,
                "mean_a": res.mean_a, "mean_b": res.mean_b,
                "diff_a_minus_b": res.diff_a_minus_b,
                "pct_lift_vs_b": res.pct_lift_vs_b,
                "ci_low": res.ci_low, "ci_high": res.ci_high,
                "p_value": res.p_value,
                "effect_size": res.cohen_d,
            })
            pvals.append(float(res.p_value))

    p_adj = adjust_pvalues(np.asarray(pvals), method=p_adjust)
    for row, pa in zip(rows, p_adj):
        row["p_adj"] = float(pa)
        row["significant"] = bool(pa < alpha)

    return pd.DataFrame(rows).sort_values("p_adj")


# ---------------------------------------------------------------------------
# Sample Ratio Mismatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SRMResult:
    observed: dict[str, int]
    expected_ratio: dict[str, float]
    chi2: float
    p_value: float
    mismatch: bool


def srm_check(
    observed_counts: dict[str, int],
    expected_ratio: dict[str, float] | None = None,
    alpha: float = 0.05,
) -> SRMResult:
    """Chi-square goodness-of-fit test for sample ratio mismatch."""
    names = list(observed_counts.keys())
    obs = np.array([observed_counts[n] for n in names], dtype=float)
    total = obs.sum()

    if expected_ratio is None:
        expected_ratio = {n: 1.0 / len(names) for n in names}

    ratios = np.array([expected_ratio[n] for n in names])
    ratios = ratios / ratios.sum()
    exp = total * ratios

    chi2, p_value = stats.chisquare(obs, f_exp=exp)

    return SRMResult(
        observed=observed_counts,
        expected_ratio={n: float(r) for n, r in zip(names, ratios)},
        chi2=float(chi2),
        p_value=float(p_value),
        mismatch=bool(p_value < alpha),
    )

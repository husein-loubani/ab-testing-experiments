"""
reporting.py
Manager-facing report builders for A/B test results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

from ab_testing.config import ALPHA, RANDOM_SEED, FAST_FOOD
from ab_testing.stats import (
    GlobalTestResult,
    choose_global_test,
    welch_ttest_mean,
    adjust_pvalues,
    bootstrap_ci_mean_diff,
)


@dataclass(frozen=True)
class FastFoodManagerReport:
    recommendation_md: str
    group_summary: pd.DataFrame
    pairwise_summary: pd.DataFrame
    global_test: GlobalTestResult
    posthoc: Optional[pd.DataFrame]


def _k(x: float) -> str:
    return f"{x:.2f}k"


def build_fast_food_manager_report(
    store_level: pd.DataFrame,
    *,
    promotion_col: str = FAST_FOOD.col_promotion,
    metric_col: str = FAST_FOOD.primary_metric,
    alpha: float = ALPHA,
    seed: int = RANDOM_SEED,
    p_adjust: Literal["holm", "fdr_bh"] = "holm",
) -> FastFoodManagerReport:
    grp = (
        store_level.groupby(promotion_col, as_index=False)
        .agg(
            stores=(metric_col, "size"),
            avg_weekly_sales_k=(metric_col, "mean"),
            median_k=(metric_col, "median"),
            std_k=(metric_col, "std"),
        )
        .sort_values(promotion_col)
    )

    means = grp.set_index(promotion_col)["avg_weekly_sales_k"]

    groups = {
        f"Promotion {p}": store_level.loc[store_level[promotion_col] == p, metric_col].to_numpy()
        for p in sorted(store_level[promotion_col].unique())
    }
    global_res = choose_global_test(groups, alpha=alpha)

    names = list(groups.keys())
    rows = []
    pvals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            x, y = groups[a], groups[b]
            res = welch_ttest_mean(x, y, group_a=a, group_b=b, alpha=alpha)
            b_lo, b_hi = bootstrap_ci_mean_diff(x, y, alpha=alpha, n_boot=20000, seed=seed)
            rows.append({
                "comparison": f"{a} vs {b}",
                "n_a": res.n_a, "n_b": res.n_b,
                "mean_a": res.mean_a, "mean_b": res.mean_b,
                "diff_a_minus_b": res.diff_a_minus_b,
                "pct_lift_vs_b": res.pct_lift_vs_b,
                "ci_low": res.ci_low, "ci_high": res.ci_high,
                "bootstrap_ci_low": b_lo, "bootstrap_ci_high": b_hi,
                "p_value": res.p_value,
            })
            pvals.append(res.p_value)

    p_adj = adjust_pvalues(np.asarray(pvals), method=p_adjust)
    for row, pa in zip(rows, p_adj):
        row["p_adj"] = float(pa)
        row["significant"] = bool(pa < alpha)

    pairwise = pd.DataFrame(rows).sort_values("p_adj")
    posthoc = pairwise.copy() if global_res.p_value < alpha else None

    top_promo = int(means.idxmax())
    winner = f"Promotion {top_promo}"

    ordered_pairs = [("Promotion 1", "Promotion 2"), ("Promotion 1", "Promotion 3"), ("Promotion 2", "Promotion 3")]
    bullets = []
    for a_name, b_name in ordered_pairs:
        r = pairwise[pairwise["comparison"].isin([f"{a_name} vs {b_name}", f"{b_name} vs {a_name}"])]
        if r.empty:
            continue
        row = r.iloc[0]
        bullets.append(
            f"- {row['comparison']}: lift {_k(float(row['diff_a_minus_b']))}, "
            f"95% CI [{_k(float(row['ci_low']))}, {_k(float(row['ci_high']))}], "
            f"adjusted p-value {float(row['p_adj']):.3f}"
        )

    if global_res.p_value < alpha:
        global_line = f"A/B/C global test is significant ({global_res.method}, p = {global_res.p_value:.3f}). Holm correction used for pairwise comparisons."
    else:
        global_line = f"A/B/C global test is not significant at alpha {alpha} ({global_res.method}, p = {global_res.p_value:.3f}). Pairwise results shown with Holm correction for transparency."

    rec = "\n".join([
        "### Decision",
        f"**Recommendation:** Roll out **{winner}** based on the highest average store-level weekly sales.",
        "",
        "### Evidence",
        global_line,
        "",
        "Pairwise results (Holm-corrected):",
        *bullets,
        "",
        "### Next steps",
        "- Confirm promotion cost and margin impact (sales alone may not maximize profit)",
        "- If differences are small or not significant, run a longer test or add more stores",
        "- Monitor rollout with guardrail metrics (operational load, customer satisfaction)",
    ])

    pairwise_summary = pairwise[
        ["comparison", "n_a", "n_b", "mean_a", "mean_b", "pct_lift_vs_b", "ci_low", "ci_high", "p_adj", "significant"]
    ].copy()

    return FastFoodManagerReport(
        recommendation_md=rec,
        group_summary=grp,
        pairwise_summary=pairwise_summary,
        global_test=global_res,
        posthoc=posthoc,
    )

"""
plotting.py
All reusable visualisation functions for the A/B Testing project.
- Matplotlib: individual static charts (saved as PNG)
- Plotly: interactive executive dashboards (saved as HTML)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ab_testing.config import FIGURES_FF_DIR, FIGURES_CC_DIR
from ab_testing.io import ensure_parent_dir


ACCENT_A = "#2980b9"
ACCENT_B = "#c0392b"


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11


def save_fig(fig: plt.Figure, path: Path, dpi: int = 160) -> None:
    ensure_parent_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _ordered_categories(values: pd.Series, order: list[Any] | None) -> list[Any]:
    if order is not None:
        return list(order)
    return sorted(values.dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Static Matplotlib plots: Fast Food
# ---------------------------------------------------------------------------

def plot_group_mean_bar(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    out_name: str,
    order: list[Any] | None = None,
) -> plt.Figure:
    cats = _ordered_categories(df[group_col], order)

    stats_df = (
        df[df[group_col].isin(cats)]
        .groupby(group_col, as_index=False)
        .agg(mean=(value_col, "mean"), n=(value_col, "size"))
    )
    stats_df[group_col] = pd.Categorical(stats_df[group_col], categories=cats, ordered=True)
    stats_df = stats_df.sort_values(group_col)

    labels = [f"{g}\nn={n}" for g, n in zip(stats_df[group_col].tolist(), stats_df["n"].tolist())]
    means = stats_df["mean"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(means)), means, color=ACCENT_A, edgecolor="white")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    save_fig(fig, FIGURES_FF_DIR / out_name)
    return fig


def plot_weekly_trend(
    df: pd.DataFrame,
    *,
    week_col: str,
    group_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    out_name: str,
    order: list[Any] | None = None,
) -> plt.Figure:
    groups = _ordered_categories(df[group_col], order)

    weekly = (
        df[df[group_col].isin(groups)]
        .groupby([week_col, group_col], as_index=False)
        .agg(mean_value=(value_col, "mean"))
        .sort_values([group_col, week_col])
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    for g in groups:
        sub = weekly[weekly[group_col] == g]
        ax.plot(sub[week_col], sub["mean_value"], marker="o", label=str(g))
        for x, y in zip(sub[week_col].tolist(), sub["mean_value"].tolist()):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel(str(week_col))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=group_col)

    save_fig(fig, FIGURES_FF_DIR / out_name)
    return fig


def plot_distribution_violin(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_name: str,
    figures_dir: Path | None = None,
    order: list[Any] | None = None,
) -> plt.Figure:
    cats = _ordered_categories(df[group_col], order)
    dest = figures_dir or FIGURES_FF_DIR

    data: list[np.ndarray] = []
    ns: list[int] = []
    for c in cats:
        vals = pd.to_numeric(df.loc[df[group_col] == c, value_col], errors="coerce").dropna().to_numpy()
        data.append(vals)
        ns.append(int(vals.size))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(cats, ns)])

    means = [float(np.mean(v)) if v.size else float("nan") for v in data]
    for i, m in enumerate(means, start=1):
        if np.isfinite(m):
            ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    save_fig(fig, dest / out_name)
    return fig


def plot_retention_rate_bar(
    df: pd.DataFrame,
    *,
    group_col: str,
    retention_col: str,
    title: str,
    ylabel: str,
    out_name: str,
    order: list[str] | None = None,
) -> plt.Figure:
    cats = _ordered_categories(df[group_col], order)

    tmp = df[df[group_col].isin(cats)].copy()
    tmp[retention_col] = pd.to_numeric(tmp[retention_col], errors="coerce")

    agg = (
        tmp.groupby(group_col, as_index=False)
        .agg(rate=(retention_col, "mean"), n=(retention_col, "size"))
    )
    agg[group_col] = pd.Categorical(agg[group_col], categories=cats, ordered=True)
    agg = agg.sort_values(group_col)

    labels = [f"{g}\nn={n}" for g, n in zip(agg[group_col].tolist(), agg["n"].tolist())]
    rates = agg["rate"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(rates)), rates, color=ACCENT_A, edgecolor="white")
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)

    for i, r in enumerate(rates):
        if np.isfinite(r):
            ax.text(i, r, f"{r:.1%}", ha="center", va="bottom", fontsize=9)

    save_fig(fig, FIGURES_CC_DIR / out_name)
    return fig


# ---------------------------------------------------------------------------
# Interactive Plotly Dashboard: Fast Food
# ---------------------------------------------------------------------------

def dashboard_fast_food(
    df_clean: pd.DataFrame,
    store_level: pd.DataFrame,
    *,
    promotion_col: str = "Promotion",
    sales_col: str = "SalesInThousands",
    week_col: str = "week",
    out_name: str = "dashboard_fast_food.html",
) -> "plotly.graph_objects.Figure":
    """Interactive dark-themed executive dashboard for the Fast Food A/B test."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    promos = sorted(store_level[promotion_col].unique())
    colors = {1: "#7c3aed", 2: "#3b82f6", 3: "#10b981"}
    color_list = [colors.get(p, "#888") for p in promos]

    # ---- KPI calculations ----
    n_stores = store_level.shape[0]
    n_weeks = int(df_clean[week_col].nunique())
    best_promo = store_level.groupby(promotion_col)["avg_sales"].mean().idxmax()
    best_mean = store_level.groupby(promotion_col)["avg_sales"].mean().max()
    overall_mean = store_level["avg_sales"].mean()

    # ---- Weekly trend data ----
    weekly = (
        df_clean.groupby([week_col, promotion_col], as_index=False)
        .agg(mean_sales=(sales_col, "mean"))
        .sort_values([promotion_col, week_col])
    )

    # ---- Store-level means ----
    promo_stats = store_level.groupby(promotion_col).agg(
        mean_sales=("avg_sales", "mean"),
        std_sales=("avg_sales", "std"),
        n=("avg_sales", "size"),
    ).reindex(promos)

    # ---- Build subplots ----
    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.08, 0.46, 0.46],
        column_widths=[0.5, 0.5],
        specs=[
            [{"colspan": 2, "type": "domain"}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=[
            "",
            "Weekly Sales Trend by Promotion",
            "Store-Level Average Weekly Sales",
            "Sales Distribution by Promotion",
            "Pairwise Lift Summary",
        ],
    )

    # ---- Row 1: KPI cards as annotation (no trace needed) ----
    # We use invisible scatter to occupy the domain, then add annotations
    fig.add_trace(
        go.Scatter(x=[None], y=[None], showlegend=False),
        row=2, col=1,
    )

    # ---- Row 2, Col 1: Weekly trend (line chart) ----
    for p in promos:
        sub = weekly[weekly[promotion_col] == p]
        fig.add_trace(
            go.Scatter(
                x=sub[week_col],
                y=sub["mean_sales"],
                mode="lines+markers+text",
                name=f"Promo {p}",
                text=[f"{v:.1f}" for v in sub["mean_sales"]],
                textposition="top center",
                textfont=dict(size=10),
                line=dict(color=colors.get(p, "#888"), width=3),
                marker=dict(size=8),
            ),
            row=2, col=1,
        )
    fig.update_xaxes(title_text="Week", row=2, col=1, dtick=1)
    fig.update_yaxes(title_text="Avg Sales ($k)", row=2, col=1)

    # ---- Row 2, Col 2: Bar chart (store-level mean) ----
    fig.add_trace(
        go.Bar(
            x=[f"Promo {p}" for p in promos],
            y=promo_stats["mean_sales"].values,
            marker_color=color_list,
            text=[f"${v:.2f}k<br>n={promo_stats.loc[p, 'n']}" for p, v in zip(promos, promo_stats["mean_sales"])],
            textposition="outside",
            textfont=dict(size=11),
            showlegend=False,
        ),
        row=2, col=2,
    )
    fig.update_yaxes(title_text="Avg Weekly Sales ($k)", row=2, col=2)

    # ---- Row 3, Col 1: Box plot (distribution) ----
    for p in promos:
        vals = store_level.loc[store_level[promotion_col] == p, "avg_sales"]
        fig.add_trace(
            go.Box(
                y=vals,
                name=f"Promo {p}",
                marker_color=colors.get(p, "#888"),
                boxmean=True,
                showlegend=False,
            ),
            row=3, col=1,
        )
    fig.update_yaxes(title_text="Avg Weekly Sales ($k)", row=3, col=1)

    # ---- Row 3, Col 2: Pairwise lift table ----
    pairs = [(1, 2), (1, 3), (2, 3)]
    means_dict = promo_stats["mean_sales"].to_dict()
    comparisons, lifts_k, lifts_pct = [], [], []
    for a, b in pairs:
        ma, mb = means_dict[a], means_dict[b]
        lift = ma - mb
        lift_p = lift / mb * 100
        comparisons.append(f"Promo {a} vs {b}")
        lifts_k.append(f"{lift:+.2f}")
        lifts_pct.append(f"{lift_p:+.1f}%")

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Comparison</b>", "<b>Lift ($k)</b>", "<b>Lift (%)</b>"],
                fill_color="#1e1b4b",
                font=dict(color="white", size=13),
                align="center",
                height=35,
            ),
            cells=dict(
                values=[comparisons, lifts_k, lifts_pct],
                fill_color=[["#2d2a5e"] * len(comparisons)],
                font=dict(color="white", size=12),
                align="center",
                height=30,
            ),
        ),
        row=3, col=2,
    )

    # ---- Layout: dark theme matching the reference ----
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f0e1a",
        plot_bgcolor="#1a1933",
        font=dict(family="Inter, system-ui, sans-serif", color="#e2e8f0", size=13),
        title=dict(
            text=(
                "<b>Fast Food Marketing Campaign: A/B Test Dashboard</b>"
                f"<br><span style='font-size:13px; color:#94a3b8'>"
                f"Stores: {n_stores} | Weeks: {n_weeks} | "
                f"Best Promotion: {best_promo} (${best_mean:.2f}k avg) | "
                f"Overall Mean: ${overall_mean:.2f}k</span>"
            ),
            font=dict(size=18, color="#a78bfa"),
            x=0.5,
            xanchor="center",
        ),
        height=950,
        margin=dict(t=100, b=40, l=60, r=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color="#c4b5fd")

    out_path = FIGURES_FF_DIR / out_name
    ensure_parent_dir(out_path)
    fig.write_html(str(out_path), include_plotlyjs=True)
    return fig


# ---------------------------------------------------------------------------
# Interactive Plotly Dashboard: Cookie Cats
# ---------------------------------------------------------------------------

def dashboard_cookie_cats(
    df: pd.DataFrame,
    *,
    version_col: str = "version",
    ret1_col: str = "retention_1",
    ret7_col: str = "retention_7",
    rounds_col: str = "sum_gamerounds",
    out_name: str = "dashboard_cookie_cats.html",
) -> "plotly.graph_objects.Figure":
    """Interactive dark-themed executive dashboard for the Cookie Cats A/B test."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    groups = ["gate_30", "gate_40"]
    colors_map = {"gate_30": "#7c3aed", "gate_40": "#3b82f6"}

    # ---- KPI calculations ----
    n_players = len(df)
    ret7_overall = df[ret7_col].mean()
    ret1_overall = df[ret1_col].mean()
    avg_rounds = df[rounds_col].mean()

    ret_data = df.groupby(version_col).agg(
        day1=(ret1_col, "mean"),
        day7=(ret7_col, "mean"),
        avg_rounds=(rounds_col, "mean"),
        median_rounds=(rounds_col, "median"),
        n=(ret1_col, "size"),
    ).reindex(groups)

    # ---- Build subplots ----
    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.33, 0.33, 0.34],
        column_widths=[0.5, 0.5],
        specs=[
            [{"type": "xy"}, {"type": "table"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Retention: Day 1 vs Day 7",
            "Treatment Effect Summary",
            "Engagement Distribution (log scale)",
            "Retention Funnel",
            "",
            "Group Statistics",
        ],
    )

    # ---- Row 1, Col 1: Grouped bar chart (retention Day 1 + Day 7) ----
    x_labels = [f"{g} (n={ret_data.loc[g, 'n']:,.0f})" for g in groups]

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=ret_data["day1"].values,
            name="Day 1 Retention",
            marker_color="#3b82f6",
            text=[f"{v:.1%}" for v in ret_data["day1"]],
            textposition="outside",
            textfont=dict(size=11),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=ret_data["day7"].values,
            name="Day 7 Retention",
            marker_color="#ef4444",
            text=[f"{v:.1%}" for v in ret_data["day7"]],
            textposition="outside",
            textfont=dict(size=11),
        ),
        row=1, col=1,
    )
    fig.update_yaxes(title_text="Retention Rate", range=[0, 0.6], row=1, col=1)

    # ---- Row 1, Col 2: Treatment effect table ----
    metrics_names = ["Day 1 Retention", "Day 7 Retention", "Avg Game Rounds"]
    gate30_vals, gate40_vals, diffs_abs, diffs_rel = [], [], [], []

    for col, fmt in [(ret1_col, ".2%"), (ret7_col, ".2%"), (rounds_col, ".1f")]:
        r30 = df[df[version_col] == "gate_30"][col].mean()
        r40 = df[df[version_col] == "gate_40"][col].mean()
        abs_d = r30 - r40
        rel_d = abs_d / r40 * 100 if r40 != 0 else 0

        if "%" in fmt:
            gate30_vals.append(f"{r30:{fmt}}")
            gate40_vals.append(f"{r40:{fmt}}")
            diffs_abs.append(f"{abs_d * 100:+.2f} pp")
        else:
            gate30_vals.append(f"{r30:{fmt}}")
            gate40_vals.append(f"{r40:{fmt}}")
            diffs_abs.append(f"{abs_d:+.1f}")
        diffs_rel.append(f"{rel_d:+.1f}%")

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>gate_30</b>", "<b>gate_40</b>", "<b>Diff</b>", "<b>Rel. Diff</b>"],
                fill_color="#1e1b4b",
                font=dict(color="white", size=12),
                align="center",
                height=32,
            ),
            cells=dict(
                values=[metrics_names, gate30_vals, gate40_vals, diffs_abs, diffs_rel],
                fill_color=[["#2d2a5e"] * len(metrics_names)],
                font=dict(color="white", size=11),
                align="center",
                height=28,
            ),
        ),
        row=1, col=2,
    )

    # ---- Row 2, Col 1: Engagement violin (log scale) ----
    for g in groups:
        vals = np.log1p(df.loc[df[version_col] == g, rounds_col].values)
        fig.add_trace(
            go.Violin(
                y=vals,
                name=g,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors_map[g],
                opacity=0.7,
                line_color="white",
                showlegend=False,
            ),
            row=2, col=1,
        )
    fig.update_yaxes(title_text="log1p(gamerounds)", row=2, col=1)

    # ---- Row 2, Col 2: Retention funnel (horizontal bar) ----
    for g in groups:
        sub = df[df[version_col] == g]
        n_total = len(sub)
        n_d1 = int(sub[ret1_col].sum())
        n_d7 = int(sub[ret7_col].sum())
        rates = [1.0, n_d1 / n_total, n_d7 / n_total]

        fig.add_trace(
            go.Bar(
                y=["Install", "Day 1", "Day 7"],
                x=rates,
                orientation="h",
                name=g,
                marker_color=colors_map[g],
                opacity=0.8,
                text=[f"{r:.1%}" for r in rates],
                textposition="outside",
                textfont=dict(size=10),
                showlegend=False,
            ),
            row=2, col=2,
        )
    fig.update_xaxes(title_text="Rate", range=[0, 1.2], row=2, col=2)
    fig.update_layout(barmode="group")

    # ---- Row 3, Col 2: Group statistics table ----
    stat_groups = groups
    stat_n = [f"{ret_data.loc[g, 'n']:,.0f}" for g in stat_groups]
    stat_ret1 = [f"{ret_data.loc[g, 'day1']:.2%}" for g in stat_groups]
    stat_ret7 = [f"{ret_data.loc[g, 'day7']:.2%}" for g in stat_groups]
    stat_avg = [f"{ret_data.loc[g, 'avg_rounds']:.1f}" for g in stat_groups]
    stat_med = [f"{ret_data.loc[g, 'median_rounds']:.0f}" for g in stat_groups]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Group</b>", "<b>N</b>", "<b>Day 1 Ret.</b>", "<b>Day 7 Ret.</b>",
                        "<b>Avg Rounds</b>", "<b>Med Rounds</b>"],
                fill_color="#1e1b4b",
                font=dict(color="white", size=12),
                align="center",
                height=32,
            ),
            cells=dict(
                values=[stat_groups, stat_n, stat_ret1, stat_ret7, stat_avg, stat_med],
                fill_color=[["#2d2a5e"] * len(stat_groups)],
                font=dict(color="white", size=11),
                align="center",
                height=28,
            ),
        ),
        row=3, col=2,
    )

    # ---- Layout: dark theme ----
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f0e1a",
        plot_bgcolor="#1a1933",
        font=dict(family="Inter, system-ui, sans-serif", color="#e2e8f0", size=13),
        title=dict(
            text=(
                "<b>Cookie Cats: A/B Test Dashboard (gate_30 vs gate_40)</b>"
                f"<br><span style='font-size:13px; color:#94a3b8'>"
                f"Players: {n_players:,} | "
                f"Day 1 Retention: {ret1_overall:.1%} | "
                f"Day 7 Retention: {ret7_overall:.1%} | "
                f"Avg Rounds: {avg_rounds:.1f}</span>"
            ),
            font=dict(size=18, color="#a78bfa"),
            x=0.5,
            xanchor="center",
        ),
        height=1050,
        margin=dict(t=100, b=40, l=60, r=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color="#c4b5fd")

    out_path = FIGURES_CC_DIR / out_name
    ensure_parent_dir(out_path)
    fig.write_html(str(out_path), include_plotlyjs=True)
    return fig

"""
plotting.py
All reusable Matplotlib visualisation functions for the A/B Testing project.
Every function returns a Figure without calling plt.show().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ab_testing.config import FIGURES_DIR
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

    save_fig(fig, FIGURES_DIR / out_name)
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

    save_fig(fig, FIGURES_DIR / out_name)
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
    order: list[Any] | None = None,
) -> plt.Figure:
    cats = _ordered_categories(df[group_col], order)

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

    save_fig(fig, FIGURES_DIR / out_name)
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

    save_fig(fig, FIGURES_DIR / out_name)
    return fig


# ---------------------------------------------------------------------------
# Dashboard: Fast Food
# ---------------------------------------------------------------------------

def dashboard_fast_food(
    df_clean: pd.DataFrame,
    store_level: pd.DataFrame,
    *,
    promotion_col: str = "Promotion",
    sales_col: str = "SalesInThousands",
    week_col: str = "week",
    out_name: str = "dashboard_fast_food.png",
) -> plt.Figure:
    """Four-panel executive dashboard for the Fast Food A/B test."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Fast Food Marketing Campaign — A/B Test Dashboard", fontsize=16, y=0.98)

    colors = {1: "#2980b9", 2: "#e67e22", 3: "#27ae60"}
    promos = sorted(store_level[promotion_col].unique())

    # --- Panel 1: Weekly sales trend by promotion ---
    ax = axes[0, 0]
    weekly = (
        df_clean.groupby([week_col, promotion_col], as_index=False)
        .agg(mean_sales=(sales_col, "mean"))
        .sort_values([promotion_col, week_col])
    )
    for p in promos:
        sub = weekly[weekly[promotion_col] == p]
        ax.plot(sub[week_col], sub["mean_sales"], marker="o", color=colors[p],
                label=f"Promo {p}", linewidth=2, markersize=7)
        for x_val, y_val in zip(sub[week_col], sub["mean_sales"]):
            ax.text(x_val, y_val + 0.15, f"{y_val:.1f}", ha="center", fontsize=7.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Avg Sales (thousands $)")
    ax.set_title("Weekly Sales Trend by Promotion")
    ax.legend(fontsize=9)
    ax.set_xticks(sorted(df_clean[week_col].unique()))

    # --- Panel 2: Store-level mean bar chart ---
    ax = axes[0, 1]
    means = store_level.groupby(promotion_col)["avg_sales"].mean().reindex(promos)
    ns = store_level.groupby(promotion_col)["avg_sales"].size().reindex(promos)
    bar_colors = [colors[p] for p in promos]
    ax.bar(range(len(promos)), means.values, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(promos)))
    ax.set_xticklabels([f"Promo {p}\nn={ns[p]}" for p in promos])
    ax.set_ylabel("Avg Weekly Sales (thousands $)")
    ax.set_title("Store-Level Average Weekly Sales")
    for i, m in enumerate(means.values):
        ax.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")

    # --- Panel 3: Violin distribution ---
    ax = axes[1, 0]
    data = [store_level.loc[store_level[promotion_col] == p, "avg_sales"].values for p in promos]
    parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    for i, pc in enumerate(parts.get("bodies", [])):
        pc.set_facecolor(bar_colors[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(promos) + 1))
    ax.set_xticklabels([f"Promo {p}" for p in promos])
    ax.set_ylabel("Avg Weekly Sales (thousands $)")
    ax.set_title("Sales Distribution by Promotion")
    group_means = [np.mean(d) for d in data]
    for i, m in enumerate(group_means, start=1):
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    # --- Panel 4: Pairwise lift summary ---
    ax = axes[1, 1]
    ax.axis("off")
    pairs = [(1, 2), (1, 3), (2, 3)]
    table_data = [["Comparison", "Lift ($k)", "Lift (%)"]]
    for a, b in pairs:
        ma = means[a]
        mb = means[b]
        lift_k = ma - mb
        lift_pct = lift_k / mb * 100
        table_data.append([f"Promo {a} vs {b}", f"{lift_k:+.2f}", f"{lift_pct:+.1f}%"])

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if row % 2 == 0 else "white")
    ax.set_title("Pairwise Lift Summary", pad=20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, FIGURES_DIR / out_name)
    return fig


# ---------------------------------------------------------------------------
# Dashboard: Cookie Cats
# ---------------------------------------------------------------------------

def dashboard_cookie_cats(
    df: pd.DataFrame,
    *,
    version_col: str = "version",
    ret1_col: str = "retention_1",
    ret7_col: str = "retention_7",
    rounds_col: str = "sum_gamerounds",
    out_name: str = "dashboard_cookie_cats.png",
) -> plt.Figure:
    """Four-panel executive dashboard for the Cookie Cats A/B test."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Cookie Cats — A/B Test Dashboard (gate_30 vs gate_40)", fontsize=16, y=0.98)

    colors_map = {"gate_30": "#2980b9", "gate_40": "#c0392b"}
    groups = ["gate_30", "gate_40"]

    # --- Panel 1: Retention comparison (Day 1 + Day 7 grouped bar) ---
    ax = axes[0, 0]
    ret_data = df.groupby(version_col).agg(
        day1=(ret1_col, "mean"), day7=(ret7_col, "mean"), n=(ret1_col, "size")
    ).reindex(groups)

    x_pos = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x_pos - width / 2, ret_data["day1"], width, label="Day 1",
                   color="#3498db", edgecolor="white")
    bars7 = ax.bar(x_pos + width / 2, ret_data["day7"], width, label="Day 7",
                   color="#e74c3c", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{g}\nn={ret_data.loc[g, 'n']:,}" for g in groups])
    ax.set_ylabel("Retention Rate")
    ax.set_title("Retention: Day 1 vs Day 7")
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=9)
    for bars in [bars1, bars7]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.1%}", ha="center", fontsize=8.5, fontweight="bold")

    # --- Panel 2: Treatment effect summary table ---
    ax = axes[0, 1]
    ax.axis("off")
    metrics = [
        ("Day 1 retention", ret1_col),
        ("Day 7 retention", ret7_col),
    ]
    table_data = [["Metric", "gate_30", "gate_40", "Diff (pp)", "Rel. Diff"]]
    for label, col in metrics:
        r30 = df[df[version_col] == "gate_30"][col].mean()
        r40 = df[df[version_col] == "gate_40"][col].mean()
        abs_d = (r30 - r40) * 100
        rel_d = (r30 - r40) / r40 * 100 if r40 > 0 else 0
        table_data.append([label, f"{r30:.2%}", f"{r40:.2%}", f"{abs_d:+.2f}", f"{rel_d:+.1f}%"])

    m30 = df[df[version_col] == "gate_30"][rounds_col].mean()
    m40 = df[df[version_col] == "gate_40"][rounds_col].mean()
    abs_e = m30 - m40
    rel_e = abs_e / m40 * 100 if m40 > 0 else 0
    table_data.append(["Avg rounds", f"{m30:.1f}", f"{m40:.1f}", f"{abs_e:+.1f}", f"{rel_e:+.1f}%"])

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if row % 2 == 0 else "white")
    ax.set_title("Treatment Effect Summary", pad=20)

    # --- Panel 3: Engagement violin (log scale) ---
    ax = axes[1, 0]
    data = [np.log1p(df.loc[df[version_col] == g, rounds_col].values) for g in groups]
    parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    for i, pc in enumerate(parts.get("bodies", [])):
        pc.set_facecolor(list(colors_map.values())[i])
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_ylabel("log1p(gamerounds)")
    ax.set_title("Engagement Distribution (log scale)")
    for i, d in enumerate(data, start=1):
        m = np.mean(d)
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    # --- Panel 4: Retention funnel ---
    ax = axes[1, 1]
    for i, g in enumerate(groups):
        sub = df[df[version_col] == g]
        n_total = len(sub)
        n_d1 = int(sub[ret1_col].sum())
        n_d7 = int(sub[ret7_col].sum())
        rates = [1.0, n_d1 / n_total, n_d7 / n_total]

        offset = -0.2 + i * 0.4
        bars = ax.barh(
            [j + offset for j in range(3)], rates,
            height=0.35, color=colors_map[g], alpha=0.8, label=g,
        )
        for j, (bar, r) in enumerate(zip(bars, rates)):
            ax.text(r + 0.01, j + offset, f"{r:.1%}", va="center", fontsize=8.5)

    ax.set_yticks(range(3))
    ax.set_yticklabels(["Install", "Day 1 Retained", "Day 7 Retained"])
    ax.set_xlabel("Rate")
    ax.set_title("Retention Funnel")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.15)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, FIGURES_DIR / out_name)
    return fig

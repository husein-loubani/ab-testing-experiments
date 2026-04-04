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

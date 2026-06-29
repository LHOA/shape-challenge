"""Visualisation helpers for the Shape DS Challenge.

Provides matplotlib/seaborn-based plotting functions for distributions,
time series, correlation heatmaps and failure analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.figure import Figure

# Global style setup
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {"ok": "#2ecc71", "fail": "#e74c3c", "pre_fail": "#f39c12"}


def plot_sensor_distributions(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> Figure:
    """Histogram of each sensor, coloured by fail state.

    Args:
        df: DataFrame with sensor columns + 'Fail'.
        save_path: If provided, saves the figure to this path.

    Returns:
        The matplotlib Figure object.

    """
    sensor_cols = [
        "Temperature",
        "Pressure",
        "VibrationX",
        "VibrationY",
        "VibrationZ",
        "Frequency",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(sensor_cols):
        ax = axes[idx]
        for state, colour in [(False, COLORS["ok"]), (True, COLORS["fail"])]:
            subset = df[df["Fail"] == state][col].dropna()
            ax.hist(
                subset,
                bins=40,
                alpha=0.6,
                color=colour,
                label=f"{'Fail' if state else 'Normal'} (n={len(subset)})",
                density=True,
            )
        ax.set_title(col, fontsize=13, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Sensor Distributions by Fail State", fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> Figure:
    """Correlation heatmap of all numeric columns.

    Args:
        df: DataFrame with numeric columns.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.

    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_time_series_with_failures(
    df: pd.DataFrame,
    sensor_col: str = "Temperature",
    save_path: str | None = None,
) -> Figure:
    """Plot a sensor's time series with failure regions highlighted.

    Args:
        df: DataFrame with 'Cycle', 'Fail' and the chosen sensor column.
        sensor_col: Column name of the sensor to plot.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.

    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df["Cycle"], df[sensor_col], color="#3498db", linewidth=0.8, alpha=0.7)

    # Highlight failure regions
    fail_mask = df["Fail"] == True  # noqa: E712
    if fail_mask.any():
        ax.scatter(
            df.loc[fail_mask, "Cycle"],
            df.loc[fail_mask, sensor_col],
            color=COLORS["fail"],
            s=20,
            label="Fail=True",
            zorder=5,
        )

    ax.set_title(f"{sensor_col} over Time — Failure Regions Highlighted", fontsize=13)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(sensor_col)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all_sensors_time_series(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> Figure:
    """Time series of all 6 sensors stacked vertically with failure highlights.

    Args:
        df: DataFrame with 'Cycle', 'Fail' and all sensor columns.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.

    """
    sensor_cols = [
        "Temperature",
        "Pressure",
        "VibrationX",
        "VibrationY",
        "VibrationZ",
        "Frequency",
    ]
    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(16, 12), sharex=True)

    for idx, col in enumerate(sensor_cols):
        ax = axes[idx]
        ax.plot(df["Cycle"], df[col], color="#3498db", linewidth=0.6, alpha=0.7)
        fail_mask = df["Fail"] == True  # noqa: E712
        if fail_mask.any():
            ax.scatter(
                df.loc[fail_mask, "Cycle"],
                df.loc[fail_mask, col],
                color=COLORS["fail"],
                s=10,
                label="Fail",
                zorder=5,
            )
        ax.set_ylabel(col, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Cycle")
    fig.suptitle("All Sensor Readings Over Time — Failure Points in Red", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pre_failure_evolution(
    pre_df: pd.DataFrame,
    save_path: str | None = None,
) -> Figure:
    """Plot average sensor evolution in the cycles before a failure event.

    Args:
        pre_df: DataFrame from `eda.pre_failure_window()` — must include
            'cycles_before' and sensor columns.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.

    """
    sensor_cols = [
        "Temperature",
        "Pressure",
        "VibrationX",
        "VibrationY",
        "VibrationZ",
        "Frequency",
    ]
    grouped = pre_df.groupby("cycles_before")[sensor_cols].mean()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(sensor_cols):
        ax = axes[idx]
        ax.plot(grouped.index, grouped[col], marker="o", color=COLORS["pre_fail"])
        ax.axvline(
            x=0, color=COLORS["fail"], linestyle="--", alpha=0.5, label="Failure"
        )
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("Cycles before failure")
        ax.set_ylabel("Mean value")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Average Sensor Behaviour Before Failure Events",
        fontsize=15,
        y=1.02,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_failure_events_timeline(
    events_df: pd.DataFrame,
    save_path: str | None = None,
) -> Figure:
    """Timeline bar chart of failure events (start cycle × duration).

    Args:
        events_df: DataFrame from `eda.identify_failure_events()`.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.

    """
    fig, ax = plt.subplots(figsize=(14, 5))

    bars = ax.bar(
        events_df["cycle_start"],
        events_df["duration_cycles"],
        width=3,
        color=COLORS["fail"],
        edgecolor="black",
        alpha=0.8,
    )
    # Label each bar with event_id
    for bar, ev_id in zip(bars, events_df["event_id"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"E{ev_id}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Cycle (event start)")
    ax.set_ylabel("Duration (cycles)")
    ax.set_title(
        f"Failure Events Timeline — {len(events_df)} events, "
        f"{events_df['duration_cycles'].sum()} total failure cycles",
        fontsize=13,
    )
    ax.set_xlim(0, 820)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_preset_failure_heatmap(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> Figure:
    """Heatmap of failure rate across (Preset_1, Preset_2) combinations.

    Args:
        df: DataFrame with 'Preset_1', 'Preset_2', 'Fail'.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.

    """
    pivot = df.pivot_table(
        index="Preset_1",
        columns="Preset_2",
        values="Fail",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot * 100,  # as percentage
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Failure Rate (%)"},
        ax=ax,
    )
    ax.set_title("Failure Rate (%) by (Preset_1, Preset_2) Configuration", fontsize=13)
    ax.set_xlabel("Preset_2")
    ax.set_ylabel("Preset_1")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

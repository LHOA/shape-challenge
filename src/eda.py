"""Exploratory Data Analysis for the Shape DS Challenge.

Provides functions to load, validate, and analyse FPSO equipment time-series data,
including failure event detection and pre-failure window extraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DATA_PATH = "data/raw/Test O_G_Equipment_Data.xlsx"
SHEET_NAME = "O&G Equipment Data"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the equipment dataset from an Excel file.

    Args:
        path: Path to the .xlsx file.

    Returns:
        DataFrame with typed columns and a datetime index placeholder (Cycle
        is kept as a sequential integer identifier).

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip()
    return df


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Run basic consistency checks on the dataset.

    Args:
        df: The raw DataFrame.

    Returns:
        A dictionary with keys:
            - n_rows: int
            - n_cols: int
            - missing: dict[str, int]
            - duplicates_cycle: int
            - dtypes: dict[str, str]
            - is_monotonic_cycle: bool

    """
    report: dict[str, Any] = {}

    report["n_rows"] = len(df)
    report["n_cols"] = len(df.columns)
    report["missing"] = df.isnull().sum().to_dict()
    report["duplicates_cycle"] = int(df["Cycle"].duplicated().sum())
    report["dtypes"] = {col: str(dt) for col, dt in df.dtypes.items()}
    report["is_monotonic_cycle"] = bool(df["Cycle"].is_monotonic_increasing)
    report["value_ranges"] = {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in df.select_dtypes(include=[np.number]).columns
        if col != "Cycle"
    }
    return report


def identify_failure_events(df: pd.DataFrame) -> pd.DataFrame:
    """Identify distinct failure events from consecutive Fail=True blocks.

    A failure *event* is a contiguous block of one or more cycles where
    Fail == True.  This is in contrast to the raw Fail column which marks
    individual cycles.

    Args:
        df: DataFrame with a 'Fail' column.

    Returns:
        DataFrame with columns:
            event_id        : 1-indexed identifier of the failure event
            cycle_start     : first cycle of the event
            cycle_end       : last cycle of the event
            duration_cycles : number of cycles the event lasted
            preset_1        : Preset_1 value at event start
            preset_2        : Preset_2 value at event start

    """
    events: list[dict[str, Any]] = []
    in_event = False
    current_event: dict[str, Any] | None = None

    for _, row in df.iterrows():
        if row["Fail"]:
            if not in_event:
                # Start of a new failure event
                current_event = {
                    "event_id": len(events) + 1,
                    "cycle_start": int(row["Cycle"]),
                    "cycle_end": int(row["Cycle"]),
                    "duration_cycles": 1,
                    "preset_1": int(row["Preset_1"]),
                    "preset_2": int(row["Preset_2"]),
                }
                in_event = True
            else:
                # Extend the current event
                assert current_event is not None
                current_event["cycle_end"] = int(row["Cycle"])
                current_event["duration_cycles"] += 1
        else:
            if in_event and current_event is not None:
                events.append(current_event)
                current_event = None
                in_event = False

    # In case the dataset ends during a failure event
    if in_event and current_event is not None:
        events.append(current_event)

    return pd.DataFrame(events)


def pre_failure_window(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Extract sensor readings from the N cycles preceding each failure event.

    Args:
        df: Full time-series DataFrame.
        events_df: Failure events DataFrame from `identify_failure_events`.
        window: Number of cycles to look back before the event start.

    Returns:
        DataFrame with one row per pre-failure cycle, annotated with
        `event_id`, `cycles_before` (negative offset) and the original
        sensor columns.

    """
    sensor_cols = [
        "Temperature",
        "Pressure",
        "VibrationX",
        "VibrationY",
        "VibrationZ",
        "Frequency",
    ]
    records: list[dict[str, Any]] = []

    for _, event in events_df.iterrows():
        start = event["cycle_start"]
        pre = df[df["Cycle"].between(start - window, start - 1)].copy()
        for _, row in pre.iterrows():
            rec = {"event_id": event["event_id"]}
            rec["cycles_before"] = int(row["Cycle"] - start)
            for col in sensor_cols:
                rec[col] = row[col]
            rec["preset_1"] = int(row["Preset_1"])
            rec["preset_2"] = int(row["Preset_2"])
            records.append(rec)

    return pd.DataFrame(records)


def describe_by_fail_state(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics grouped by fail state.

    Args:
        df: DataFrame with sensor columns + 'Fail'.

    Returns:
        DataFrame with mean, std, min, max per group (Fail=True/False)
        for each sensor column.

    """
    sensor_cols = [
        "Temperature",
        "Pressure",
        "VibrationX",
        "VibrationY",
        "VibrationZ",
        "Frequency",
    ]
    grouped = df.groupby("Fail")[sensor_cols].agg(["mean", "std", "min", "max"])
    return grouped

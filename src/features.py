"""Feature engineering for the Shape DS Challenge.

Adds lag, rolling-window, and interaction features to the equipment
time-series DataFrame for predictive modelling.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

SENSOR_COLS: list[str] = [
    "Temperature",
    "Pressure",
    "VibrationX",
    "VibrationY",
    "VibrationZ",
    "Frequency",
]


def add_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str] = SENSOR_COLS,
    lags: Sequence[int] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """Add lagged versions of the specified columns.

    Args:
        df: DataFrame sorted by 'Cycle'.
        columns: Column names to lag.
        lags: Lag values in number of cycles.

    Returns:
        DataFrame with additional columns named ``{col}_lag_{k}``.

    """
    result = df.copy()
    for col in columns:
        for k in lags:
            result[f"{col}_lag_{k}"] = result[col].shift(k)
    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: Sequence[str] = SENSOR_COLS,
    windows: Sequence[int] = (3, 5),
) -> pd.DataFrame:
    """Add rolling mean, std, and first-difference for the target columns.

    Args:
        df: DataFrame sorted by 'Cycle'.
        columns: Column names to process.
        windows: Rolling window sizes.

    Returns:
        DataFrame with additional columns:
            - ``{col}_rollmean_{w}`` — rolling mean
            - ``{col}_rollstd_{w}`` — rolling standard deviation
            - ``{col}_diff`` — first-order difference (lag-1)

    """
    result = df.copy()
    for col in columns:
        for w in windows:
            result[f"{col}_rollmean_{w}"] = (
                result[col].rolling(window=w, min_periods=1).mean()
            )
            result[f"{col}_rollstd_{w}"] = (
                result[col].rolling(window=w, min_periods=1).std().fillna(0)
            )
        # First difference (rate of change)
        result[f"{col}_diff"] = result[col].diff(1).fillna(0)
    return result


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature interactions — ratios and combined magnitudes.

    Args:
        df: DataFrame with sensor columns.

    Returns:
        DataFrame with additional columns:
            - ``Temp_Pressure_ratio`` — Temperature / Pressure
            - ``Vib_magnitude_XY`` — sqrt(VibX² + VibY²)
            - ``Vib_magnitude_XYZ`` — sqrt(VibX² + VibY² + VibZ²)

    """
    result = df.copy()
    eps = 1e-6  # avoid division by zero
    result["Temp_Pressure_ratio"] = result["Temperature"] / (result["Pressure"] + eps)
    result["Vib_magnitude_XY"] = np.sqrt(
        result["VibrationX"] ** 2 + result["VibrationY"] ** 2
    )
    result["Vib_magnitude_XYZ"] = np.sqrt(
        result["VibrationX"] ** 2
        + result["VibrationY"] ** 2
        + result["VibrationZ"] ** 2,
    )
    return result


def encode_presets(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Preset_1 and Preset_2 columns.

    Args:
        df: DataFrame with 'Preset_1' and 'Preset_2' integer columns.

    Returns:
        DataFrame with original preset columns replaced by dummies.

    """
    return pd.get_dummies(df, columns=["Preset_1", "Preset_2"], prefix=["P1", "P2"])


def shift_target_for_prediction(
    df: pd.DataFrame,
    horizon: int = 1,
    remove_failure_cycles: bool = True,
) -> pd.DataFrame:
    """Shift the Fail target forward to predict failure BEFORE it occurs.

    The new target ``Fail_pred`` is True at cycle *t* if the equipment will
    be in failure at cycle *t + horizon* (or any cycle between *t+1* and
    *t+horizon*).

    Args:
        df: DataFrame with 'Fail' column, sorted by 'Cycle'.
        horizon: How many cycles ahead to predict (default 1 = next cycle).
        remove_failure_cycles: If True, remove rows where Fail is already
            True (the model should learn to predict *onset*, not persistence).

    Returns:
        DataFrame with added ``Fail_pred`` column. If ``remove_failure_cycles``
        is True, rows where original ``Fail == True`` are dropped.

    """
    result = df.copy()

    # Fail_pred[t] = True if Fail[t+1]..Fail[t+horizon] contains any True
    result["Fail_pred"] = (
        result["Fail"]
        .rolling(window=horizon, min_periods=1)
        .max()
        .shift(-horizon)
        .fillna(0)
        .astype(bool)
    )

    if remove_failure_cycles:
        result = result[~result["Fail"]].copy()

    return result.reset_index(drop=True)


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature-engineering steps in order.

    Args:
        df: Raw DataFrame sorted by 'Cycle'.

    Returns:
        DataFrame with all engineered features added. NaN rows introduced
        by lag/rolling operations are kept and later handled by the model.

    """
    result = df.copy()
    result = add_lag_features(result)
    result = add_rolling_features(result)
    result = add_interaction_features(result)
    result = encode_presets(result)
    return result

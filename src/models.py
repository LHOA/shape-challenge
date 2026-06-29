"""Predictive models for the Shape DS Challenge.

Provides training, evaluation and interpretation of binary classifiers
for equipment failure prediction.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.figure import Figure
from xgboost import XGBClassifier

# ── Splitting ────────────────────────────────────────────────────────


def temporal_train_test_split(
    df: pd.DataFrame,
    target_col: str = "Fail",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-series-aware train/test split (no future leakage).

    Args:
        df: DataFrame sorted by 'Cycle' with all features + target.
        target_col: Name of the binary target column.
        test_size: Fraction of rows to hold out as test (last portion).

    Returns:
        Tuple (X_train, X_test, y_train, y_test).

    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    X_train = train.drop(columns=[target_col, "Cycle"])
    y_train = train[target_col].astype(int)
    X_test = test.drop(columns=[target_col, "Cycle"])
    y_test = test[target_col].astype(int)

    return X_train, X_test, y_train, y_test


# ── Baseline: Logistic Regression ────────────────────────────────────


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> LogisticRegression:
    """Train a logistic regression classifier with class balancing.

    Args:
        X_train: Training features.
        y_train: Training binary labels.
        **kwargs: Additional arguments for LogisticRegression.

    Returns:
        Fitted LogisticRegression model.

    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


# ── Main model: XGBoost ──────────────────────────────────────────────


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float | None = None,
    **kwargs: Any,
) -> XGBClassifier:
    """Train an XGBoost classifier with temporal cross-validation.

    Automatically computes ``scale_pos_weight`` if not provided
    (ratio of negative to positive samples).

    Args:
        X_train: Training features.
        y_train: Training binary labels.
        scale_pos_weight: Weight for the positive class.
        **kwargs: Additional XGBoost parameters.

    Returns:
        Fitted XGBClassifier.

    """
    if scale_pos_weight is None:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def tune_xgboost_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 3,
    scale_pos_weight: float | None = None,
) -> dict[str, Any]:
    """Simple grid search for XGBoost hyperparameters using temporal CV.

    Searches over ``max_depth``, ``learning_rate``, ``n_estimators``,
    and ``min_child_weight``.  Picks the combination with the best
    average ROC-AUC.

    Args:
        X_train: Training features.
        y_train: Training binary labels.
        cv_folds: Number of temporal CV folds.
        scale_pos_weight: Weight for the positive class.

    Returns:
        Dictionary with ``best_params`` and ``best_score``.

    """
    if scale_pos_weight is None:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / max(pos, 1)

    param_grid: dict[str, list[int | float]] = {
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200],
        "min_child_weight": [1, 3, 5],
    }

    best_score = 0.0
    best_params: dict[str, Any] = {}
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    from itertools import product

    keys = list(param_grid.keys())
    total = 1
    for v in param_grid.values():
        total *= len(v)

    print(f"Grid search: {total} combinations x {cv_folds} folds")
    idx = 0

    for values in product(*param_grid.values()):
        params = dict(zip(keys, values, strict=False))
        idx += 1

        scores: list[float] = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
            )
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_proba))

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        if idx % 9 == 0 or idx == total:
            print(
                f"  [{idx}/{total}] best so far: {best_params} (AUC={best_score:.4f})"
            )

    return {"best_params": best_params, "best_score": best_score}


# ── Temporal Cross-Validation ────────────────────────────────────────


def temporal_cv_score(
    model_class: type,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    **model_kwargs: Any,
) -> dict[str, float]:
    """Evaluate a model using time-series cross-validation.

    Args:
        model_class: Uninitialised sklearn-compatible classifier class.
        X: Full feature DataFrame (must be in temporal order).
        y: Full target Series.
        n_splits: Number of CV folds.
        **model_kwargs: Arguments forwarded to model constructor.

    Returns:
        Dictionary with mean ROC-AUC, mean F1, mean AP across folds.

    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    roc_scores: list[float] = []
    f1_scores: list[float] = []
    ap_scores: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        roc_scores.append(roc_auc_score(y_val, y_proba))
        f1_scores.append(f1_score(y_val, y_pred))
        ap_scores.append(average_precision_score(y_val, y_proba))

    return {
        "roc_auc_mean": float(np.mean(roc_scores)),
        "roc_auc_std": float(np.std(roc_scores)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "ap_mean": float(np.mean(ap_scores)),
        "ap_std": float(np.std(ap_scores)),
    }


# ── Evaluation ───────────────────────────────────────────────────────


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Generate a comprehensive evaluation report for a fitted classifier.

    Args:
        model: Fitted classifier with ``predict`` and ``predict_proba``.
        X_test: Test features.
        y_test: Test binary labels.
        model_name: Human-readable model name for titles.

    Returns:
        Dictionary with y_pred, y_proba, AUC, AP, classification report,
        and confusion matrix.

    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result: dict[str, Any] = {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation")
    print(f"{'='*50}")
    print(f"  ROC-AUC         : {result['roc_auc']:.4f}")
    print(f"  Avg Precision   : {result['avg_precision']:.4f}")
    print(f"  F1 Score        : {result['f1']:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"\n  Confusion Matrix:\n{result['confusion_matrix']}")

    return result


def plot_roc_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> Figure:
    """Plot the ROC curve.

    Args:
        y_test: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        model_name: Model name for the title.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure.

    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_precision_recall_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> Figure:
    """Plot the Precision-Recall curve (suited for imbalanced datasets).

    Args:
        y_test: True binary labels.
        y_proba: Predicted probabilities.
        model_name: Model name for the title.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure.

    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"{model_name} (AP = {ap:.4f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> Figure:
    """Plot a normalised confusion matrix.

    Args:
        y_test: True binary labels.
        y_pred: Predicted binary labels.
        model_name: Model name for the title.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure.

    """
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Fail"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format=".2f")
    ax.set_title(f"Normalised Confusion Matrix — {model_name}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Feature Importance ────────────────────────────────────────────────


def plot_xgb_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    top_n: int = 15,
    save_path: str | None = None,
) -> Figure:
    """Plot XGBoost feature importance by gain.

    Args:
        model: Fitted XGBClassifier.
        feature_names: List of all feature names used during training.
        top_n: Number of top features to display.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure.

    """
    importance = model.get_booster().get_score(importance_type="gain")
    # Map feature indices back to names
    mapped: dict[str, Any] = {}
    for key, val in importance.items():
        if key.startswith("f"):
            idx = int(key[1:])
            if idx < len(feature_names):
                mapped[feature_names[idx]] = val
            else:
                mapped[key] = val
        else:
            mapped[key] = val

    sorted_items = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*sorted_items, strict=False) if sorted_items else ([], [])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), values, color="#3498db", edgecolor="black")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"XGBoost Feature Importance (Top {top_n})")
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_summary(
    model: XGBClassifier,
    X: pd.DataFrame,
    max_display: int = 15,
    save_path: str | None = None,
) -> Figure:
    """SHAP summary (bee-swarm) plot for XGBoost.

    Args:
        model: Fitted XGBClassifier.
        X: Feature DataFrame (preferably the test set).
        max_display: Max features to show.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure.

    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title("SHAP Feature Impact Summary")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

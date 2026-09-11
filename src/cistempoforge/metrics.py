from __future__ import annotations

import numpy as np
from scipy import stats


def correlation(left, right, *, rank: bool = False) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    function = stats.spearmanr if rank else stats.pearsonr
    return float(function(left, right).statistic)


def trajectory_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    true, predicted = np.asarray(true, float), np.asarray(predicted, float)
    if true.shape != predicted.shape or true.ndim != 2 or true.shape[1] < 2:
        raise ValueError("metric inputs must be matching [genes, days] arrays")
    true_level, predicted_level = true.mean(1), predicted.mean(1)
    true_trend = true - true_level[:, None]
    predicted_trend = predicted - predicted_level[:, None]
    trajectory_p = np.asarray([
        correlation(a, b) for a, b in zip(true_trend, predicted_trend)
    ])
    trajectory_s = np.asarray([
        correlation(a, b, rank=True) for a, b in zip(true_trend, predicted_trend)
    ])
    true_endpoint = true[:, -1] - true[:, 0]
    predicted_endpoint = predicted[:, -1] - predicted[:, 0]
    daily_p = [correlation(true[:, day], predicted[:, day]) for day in range(true.shape[1])]
    return {
        "n_genes": int(len(true)),
        "n_days": int(true.shape[1]),
        "absolute_rmse": float(np.sqrt(np.mean((true - predicted) ** 2))),
        "centered_rmse": float(np.sqrt(np.mean((true_trend - predicted_trend) ** 2))),
        "median_trajectory_pearson": float(np.nanmedian(trajectory_p)),
        "mean_trajectory_pearson": float(np.nanmean(trajectory_p)),
        "median_trajectory_spearman": float(np.nanmedian(trajectory_s)),
        "level_rmse": float(np.sqrt(np.mean((true_level - predicted_level) ** 2))),
        "level_pearson": correlation(true_level, predicted_level),
        "endpoint_rmse": float(np.sqrt(np.mean((true_endpoint - predicted_endpoint) ** 2))),
        "endpoint_pearson": correlation(true_endpoint, predicted_endpoint),
        "daily_rmse": np.sqrt(np.mean((true - predicted) ** 2, axis=0)).tolist(),
        "daily_pearson": daily_p,
        "max_abs_prediction_centering_error": float(np.abs(predicted_trend.mean(1)).max()),
    }


def select_best_epoch(history: list[dict], tolerance_fraction: float = 0.005) -> dict:
    if not history:
        raise ValueError("cannot select a checkpoint from empty history")
    minimum = min(float(row["validation_centered_rmse"]) for row in history)
    threshold = minimum * (1.0 + tolerance_fraction)
    eligible = [row for row in history if float(row["validation_centered_rmse"]) <= threshold]
    def key(row):
        score = float(row["validation_median_trajectory_pearson"])
        if not np.isfinite(score):
            score = -float("inf")
        return -score, int(row["epoch"])

    return min(eligible, key=key)

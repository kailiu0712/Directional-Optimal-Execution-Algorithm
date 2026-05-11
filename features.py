from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_linear_projection(train_feature: pd.Series, train_anchor: pd.Series) -> dict[str, float]:
    train_pair = (
        pd.DataFrame({"feature": train_feature, "anchor": train_anchor})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if train_pair.empty or train_pair["anchor"].var(ddof=0) <= 1e-12:
        feature_mean = float(train_pair["feature"].mean()) if not train_pair.empty else 0.0
        anchor_mean = float(train_pair["anchor"].mean()) if not train_pair.empty else 0.0
        return {"alpha": feature_mean, "beta": 0.0, "feature_mean": feature_mean, "anchor_mean": anchor_mean}

    anchor_values = train_pair["anchor"].to_numpy(dtype=float)
    feature_values = train_pair["feature"].to_numpy(dtype=float)
    anchor_mean = float(anchor_values.mean())
    feature_mean = float(feature_values.mean())
    beta = float(
        np.mean((anchor_values - anchor_mean) * (feature_values - feature_mean))
        / max(np.var(anchor_values), 1e-12)
    )
    alpha = feature_mean - beta * anchor_mean
    return {"alpha": alpha, "beta": beta, "feature_mean": feature_mean, "anchor_mean": anchor_mean}


def _apply_linear_projection_residual(feature: pd.Series, anchor: pd.Series, state: dict[str, float]) -> pd.Series:
    projected = (
        float(state["alpha"])
        + float(state["beta"]) * anchor.replace([np.inf, -np.inf], np.nan).fillna(float(state["anchor_mean"]))
    )
    return (
        feature.replace([np.inf, -np.inf], np.nan).fillna(float(state["feature_mean"])) - projected
    ).astype(float)


def fit_augmentation_state(train_reference: pd.DataFrame) -> dict:
    return {
        "depth_slope_bid_projection": _fit_linear_projection(
            train_feature=train_reference["depth_slope_bid"],
            train_anchor=train_reference["depth_sum_bid"],
        ),
        "microprice_edge_projection": _fit_linear_projection(
            train_feature=train_reference["microprice_edge_bps"],
            train_anchor=train_reference["book_pressure_1"],
        ),
    }


def augment_feature_frame(full_history: pd.DataFrame, augmentation_state: dict) -> pd.DataFrame:
    augmented = full_history.copy()
    augmented["pressure_term_structure"] = augmented["book_pressure_1"] - augmented["book_pressure_3"]
    augmented["time_book_pressure3_interact"] = augmented["time_in_minute"] * augmented["book_pressure_3"]
    augmented["time_ask_top_share1_interact"] = augmented["time_in_minute"] * augmented["ask_top_share_1_of_5"]
    augmented["depth_slope_bid_resid"] = _apply_linear_projection_residual(
        feature=augmented["depth_slope_bid"],
        anchor=augmented["depth_sum_bid"],
        state=augmentation_state["depth_slope_bid_projection"],
    )
    augmented["microprice_edge_resid"] = _apply_linear_projection_residual(
        feature=augmented["microprice_edge_bps"],
        anchor=augmented["book_pressure_1"],
        state=augmentation_state["microprice_edge_projection"],
    )
    return augmented.replace([np.inf, -np.inf], np.nan).fillna(0.0)

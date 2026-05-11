from __future__ import annotations

import pandas as pd


def model_key(stock: str, side: str) -> str:
    return f"{stock}_{side}"


def build_second_panel(feat_df: pd.DataFrame, selected_features: list[str], top_decile: float) -> pd.DataFrame:
    second_df = feat_df.copy()
    second_df["minute"] = pd.to_datetime(second_df["minute"])
    second_df["second_in_minute"] = (second_df["time_in_minute_raw"] * 60.0).floordiv(1).clip(0, 59).astype(int)

    agg = {
        "timestamp": "last",
        "minute": "last",
        "stock": "last",
        "AskPrice_1": "mean",
        "BidPrice_1": "mean",
        "time_in_minute_raw": "mean",
    }
    for feature in selected_features:
        if feature not in agg:
            agg[feature] = "mean"

    second_panel = (
        second_df.groupby(["minute", "second_in_minute"], as_index=False)
        .agg(agg)
        .sort_values(["minute", "second_in_minute"])
        .reset_index(drop=True)
    )
    second_panel["buy_label"] = 0
    second_panel["sell_label"] = 0
    second_panel["buy_rank_pct"] = 0.0
    second_panel["sell_rank_pct"] = 0.0

    for minute, minute_df in second_panel.groupby("minute", sort=False):
        idx = minute_df.index
        buy_cutoff = float(minute_df["AskPrice_1"].quantile(float(top_decile)))
        sell_cutoff = float(minute_df["BidPrice_1"].quantile(1.0 - float(top_decile)))
        second_panel.loc[idx, "buy_label"] = (minute_df["AskPrice_1"] <= buy_cutoff).astype(int).to_numpy()
        second_panel.loc[idx, "sell_label"] = (minute_df["BidPrice_1"] >= sell_cutoff).astype(int).to_numpy()
        second_panel.loc[idx, "buy_rank_pct"] = minute_df["AskPrice_1"].rank(method="average", pct=True, ascending=True).to_numpy()
        second_panel.loc[idx, "sell_rank_pct"] = minute_df["BidPrice_1"].rank(method="average", pct=True, ascending=False).to_numpy()
    return second_panel


def ridge_minute_return_series(second_panel: pd.DataFrame, side: str) -> pd.Series:
    panel = second_panel.copy()
    future_best = pd.Series(index=panel.index, dtype=float)
    group_cols = ["minute"] if "stock" not in panel.columns else ["stock", "minute"]
    price_col = "AskPrice_1" if side == "buy" else "BidPrice_1"
    for _, minute_df in panel.groupby(group_cols, sort=False):
        price_series = minute_df[price_col]
        if side == "buy":
            best = price_series.iloc[::-1].cummin().iloc[::-1].shift(-1)
        else:
            best = price_series.iloc[::-1].cummax().iloc[::-1].shift(-1)
        future_best.loc[minute_df.index] = best.fillna(price_series)

    if side == "buy":
        denom = panel["AskPrice_1"].replace(0.0, pd.NA)
        target = 1e4 * (future_best - panel["AskPrice_1"]) / denom
    else:
        denom = panel["BidPrice_1"].replace(0.0, pd.NA)
        target = 1e4 * (panel["BidPrice_1"] - future_best) / denom
    return target.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

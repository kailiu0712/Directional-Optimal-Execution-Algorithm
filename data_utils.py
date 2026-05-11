from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BOOK_LEVELS = 5
ROLLING_WINDOWS_SECONDS = (1, 3, 10)
EPS = 1e-12
STOCK_GROUPS = {
    "AMZN_GOOG": ("AMZN", "GOOG"),
    "INTC_MSFT": ("INTC", "MSFT"),
}
STOCK_TO_GROUP = {
    stock: group_name
    for group_name, stocks in STOCK_GROUPS.items()
    for stock in stocks
}


def load_lob_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime("2024-01-01 " + df["Time"].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_div_array(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(a, default, dtype=np.float64)
    valid = np.abs(b) > EPS
    out[valid] = a[valid] / b[valid]
    return out


def _window_steps(resample_ms: int, seconds: int) -> int:
    return max(int(round(seconds * 1000 / resample_ms)), 1)


def _seconds_since_last_event(event_series: pd.Series, resample_ms: int) -> pd.Series:
    step_seconds = float(resample_ms) / 1000.0
    values = event_series.fillna(0.0).to_numpy(dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    last_idx = None
    for idx, value in enumerate(values):
        if value > 0.0:
            last_idx = idx
            out[idx] = 0.0
        elif last_idx is None:
            out[idx] = 0.0
        else:
            out[idx] = (idx - last_idx) * step_seconds
    return pd.Series(out, index=event_series.index, dtype=np.float64)


def _future_window_stat(series: pd.Series, horizon_steps: int, stat: str) -> pd.Series:
    reversed_series = series.iloc[::-1]
    if stat == "min":
        future = reversed_series.rolling(horizon_steps, min_periods=1).min().iloc[::-1]
    elif stat == "max":
        future = reversed_series.rolling(horizon_steps, min_periods=1).max().iloc[::-1]
    elif stat == "mean":
        future = reversed_series.rolling(horizon_steps, min_periods=1).mean().iloc[::-1]
    elif stat == "std":
        future = reversed_series.rolling(horizon_steps, min_periods=1).std(ddof=0).iloc[::-1].fillna(0.0)
    else:
        raise ValueError(f"Unsupported future stat: {stat}")
    return future.shift(-1)


def _coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        tmp = df.copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
        return tmp.set_index("timestamp")
    tmp = df.copy()
    tmp.index = pd.to_datetime(tmp.index)
    return tmp


def _same_group_partner(stock: str | None, other_stocks: List[str]) -> str | None:
    if stock is None:
        return other_stocks[0] if len(other_stocks) == 1 else None
    group_name = STOCK_TO_GROUP.get(stock)
    if group_name is None:
        return other_stocks[0] if len(other_stocks) == 1 else None
    for candidate in other_stocks:
        if STOCK_TO_GROUP.get(candidate) == group_name:
            return candidate
    return None


def build_features(
    df: pd.DataFrame,
    resample_ms: int = 100,
    use_log_size: bool = True,
    stock: str | None = None,
    other_stock_dfs: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if stock is not None:
        out["stock"] = stock
    out["mid"] = (out["BidPrice_1"] + out["AskPrice_1"]) / 2.0
    out["spread"] = out["AskPrice_1"] - out["BidPrice_1"]

    for i in range(1, BOOK_LEVELS + 1):
        if use_log_size:
            out[f"BidSizeLog_{i}"] = np.log1p(out[f"BidSize_{i}"])
            out[f"AskSizeLog_{i}"] = np.log1p(out[f"AskSize_{i}"])

    bid_depth_1 = out["BidSize_1"]
    ask_depth_1 = out["AskSize_1"]
    bid_depth_3 = sum(out[f"BidSize_{i}"] for i in range(1, 4))
    ask_depth_3 = sum(out[f"AskSize_{i}"] for i in range(1, 4))
    bid_depth_5 = sum(out[f"BidSize_{i}"] for i in range(1, BOOK_LEVELS + 1))
    ask_depth_5 = sum(out[f"AskSize_{i}"] for i in range(1, BOOK_LEVELS + 1))

    out["depth_sum_bid"] = bid_depth_5
    out["depth_sum_ask"] = ask_depth_5
    out["imbalance_l1"] = _safe_div(bid_depth_1 - ask_depth_1, bid_depth_1 + ask_depth_1)
    out["imbalance_l3"] = _safe_div(bid_depth_3 - ask_depth_3, bid_depth_3 + ask_depth_3)
    out["imbalance_l5"] = _safe_div(bid_depth_5 - ask_depth_5, bid_depth_5 + ask_depth_5)
    out["book_pressure_1"] = out["imbalance_l1"]
    out["book_pressure_3"] = out["imbalance_l3"]
    out["book_pressure_5"] = out["imbalance_l5"]
    out["depth_ratio_5"] = _safe_div(bid_depth_5, ask_depth_5.replace(0, np.nan))
    out["bid_top_share_1_of_5"] = _safe_div(bid_depth_1, bid_depth_5.replace(0, np.nan))
    out["ask_top_share_1_of_5"] = _safe_div(ask_depth_1, ask_depth_5.replace(0, np.nan))
    out["bid_top_share_3_of_5"] = _safe_div(bid_depth_3, bid_depth_5.replace(0, np.nan))
    out["ask_top_share_3_of_5"] = _safe_div(ask_depth_3, ask_depth_5.replace(0, np.nan))
    out["depth_concentration_diff_1_5"] = out["bid_top_share_1_of_5"] - out["ask_top_share_1_of_5"]
    out["depth_concentration_diff_3_5"] = out["bid_top_share_3_of_5"] - out["ask_top_share_3_of_5"]

    out["microprice"] = (
        out["AskPrice_1"] * out["BidSize_1"] + out["BidPrice_1"] * out["AskSize_1"]
    ) / (out["BidSize_1"] + out["AskSize_1"]).replace(0, np.nan)
    out["microprice"] = out["microprice"].fillna(out["mid"])

    tick = out["spread"].replace(0, np.nan).median()
    if pd.isna(tick) or tick == 0:
        tick = 0.01
    out["spread_bps"] = 1e4 * _safe_div(out["spread"], out["mid"].replace(0, np.nan))
    out["micro_dev"] = (out["microprice"] - out["mid"]) / tick
    out["microprice_edge_bps"] = 1e4 * _safe_div(out["microprice"] - out["mid"], out["mid"].replace(0, np.nan))
    out["depth_slope_bid"] = sum(out[f"BidSize_{i}"] / i for i in range(1, BOOK_LEVELS + 1))
    out["depth_slope_ask"] = sum(out[f"AskSize_{i}"] / i for i in range(1, BOOK_LEVELS + 1))
    out["slope_diff_5"] = out["depth_slope_bid"] - out["depth_slope_ask"]
    out["bid_up_event"] = (out["BidPrice_1"].diff().fillna(0.0) > 0.0).astype(float)
    out["bid_down_event"] = (out["BidPrice_1"].diff().fillna(0.0) < 0.0).astype(float)
    out["ask_up_event"] = (out["AskPrice_1"].diff().fillna(0.0) > 0.0).astype(float)
    out["ask_down_event"] = (out["AskPrice_1"].diff().fillna(0.0) < 0.0).astype(float)
    out["spread_change_event"] = (out["spread"].diff().abs().fillna(0.0) > EPS).astype(float)
    out["quote_price_change_event"] = (
        out["bid_up_event"] + out["bid_down_event"] + out["ask_up_event"] + out["ask_down_event"] > 0.0
    ).astype(float)
    out["bid_size_change_event"] = (out["BidSize_1"].diff().abs().fillna(0.0) > EPS).astype(float)
    out["ask_size_change_event"] = (out["AskSize_1"].diff().abs().fillna(0.0) > EPS).astype(float)
    out["book_update_event"] = (
        out["quote_price_change_event"] + out["bid_size_change_event"] + out["ask_size_change_event"] > 0.0
    ).astype(float)
    imbalance_sign = np.sign(out["imbalance_l1"].fillna(0.0))
    out["imbalance_pos_flag"] = (imbalance_sign > 0.0).astype(float)
    out["imbalance_neg_flag"] = (imbalance_sign < 0.0).astype(float)
    out["imbalance_flip_event"] = (
        pd.Series(imbalance_sign, index=out.index).diff().abs().fillna(0.0) > 0.0
    ).astype(float)

    out = out.set_index("timestamp")
    out["mid_ret_100ms"] = out["mid"].pct_change().fillna(0.0)
    out["mid_ret_500ms"] = out["mid"].pct_change(5).fillna(0.0)
    out["realized_vol_1s"] = out["mid_ret_100ms"].rolling(10, min_periods=1).std().fillna(0.0)

    trade_sign = out["Direction_1=Buy_-1=Sell"].fillna(0.0)
    trade_size = out["Size"].fillna(0.0)
    out["recent_buy_pressure"] = (trade_sign * trade_size).rolling(10, min_periods=1).sum().fillna(0.0)
    out["buy_trade_event"] = ((trade_sign > 0.0) & (trade_size > 0.0)).astype(float)
    out["sell_trade_event"] = ((trade_sign < 0.0) & (trade_size > 0.0)).astype(float)

    cancel_flag = (
        out["PartialCancel_1=Yes_0=No"].fillna(0)
        + out["FullDelete_1=Yes_0=No"].fillna(0)
    ).clip(0, 1)
    out["recent_cancel_pressure"] = cancel_flag.rolling(10, min_periods=1).sum().fillna(0.0)
    out["cancel_event"] = cancel_flag.astype(float)

    minute_floor = out.index.floor("min")
    out["time_in_minute"] = (out.index - minute_floor).total_seconds() / 60.0
    out["minute"] = minute_floor

    agg = {
        **{f"BidPrice_{i}": "last" for i in range(1, BOOK_LEVELS + 1)},
        **{f"AskPrice_{i}": "last" for i in range(1, BOOK_LEVELS + 1)},
        **{f"BidSize_{i}": "last" for i in range(1, BOOK_LEVELS + 1)},
        **{f"AskSize_{i}": "last" for i in range(1, BOOK_LEVELS + 1)},
        **{c: "last" for c in [
            "spread", "spread_bps", "mid", "microprice", "micro_dev", "microprice_edge_bps",
            "imbalance_l1", "imbalance_l3", "imbalance_l5", "book_pressure_1", "book_pressure_3", "book_pressure_5",
            "depth_sum_bid", "depth_sum_ask", "depth_ratio_5",
            "bid_top_share_1_of_5", "ask_top_share_1_of_5", "bid_top_share_3_of_5", "ask_top_share_3_of_5",
            "depth_concentration_diff_1_5", "depth_concentration_diff_3_5",
            "depth_slope_bid", "depth_slope_ask", "slope_diff_5",
            "recent_buy_pressure", "recent_cancel_pressure", "realized_vol_1s",
            "imbalance_pos_flag", "imbalance_neg_flag",
            "mid_ret_100ms", "mid_ret_500ms", "time_in_minute", "minute"
        ]},
        **{c: "sum" for c in [
            "bid_up_event", "bid_down_event", "ask_up_event", "ask_down_event",
            "spread_change_event", "quote_price_change_event", "bid_size_change_event", "ask_size_change_event",
            "book_update_event", "imbalance_flip_event", "buy_trade_event", "sell_trade_event", "cancel_event",
        ]},
    }
    if "stock" in out.columns:
        agg["stock"] = "last"
    if use_log_size:
        agg.update({f"BidSizeLog_{i}": "last" for i in range(1, BOOK_LEVELS + 1)})
        agg.update({f"AskSizeLog_{i}": "last" for i in range(1, BOOK_LEVELS + 1)})

    sampled = out.resample(f"{resample_ms}ms").agg(agg).ffill().dropna(subset=["mid"]).copy()
    sampled["minute"] = sampled.index.floor("min")

    for window_seconds in ROLLING_WINDOWS_SECONDS:
        window = _window_steps(resample_ms, window_seconds)
        label = f"{window_seconds}s"

        sampled[f"momentum_{label}_bps"] = 1e4 * sampled["mid"].pct_change(window).fillna(0.0)
        sampled[f"mid_mean_roll_{label}"] = sampled["mid"].rolling(window, min_periods=2).mean()
        sampled[f"mid_std_roll_{label}"] = sampled["mid"].rolling(window, min_periods=2).std()
        sampled[f"mid_zscore_{label}"] = _safe_div(
            sampled["mid"] - sampled[f"mid_mean_roll_{label}"],
            sampled[f"mid_std_roll_{label}"].replace(0, np.nan),
        )
        sampled[f"mid_min_roll_{label}"] = sampled["mid"].rolling(window, min_periods=2).min()
        sampled[f"mid_max_roll_{label}"] = sampled["mid"].rolling(window, min_periods=2).max()
        sampled[f"mid_range_{label}_bps"] = 1e4 * _safe_div(
            sampled[f"mid_max_roll_{label}"] - sampled[f"mid_min_roll_{label}"],
            sampled["mid"].replace(0, np.nan),
        )
        sampled[f"mid_pos_in_range_{label}"] = _safe_div(
            sampled["mid"] - sampled[f"mid_min_roll_{label}"],
            (sampled[f"mid_max_roll_{label}"] - sampled[f"mid_min_roll_{label}"]).replace(0, np.nan),
        )
        sampled[f"rv_{label}_bps"] = 1e4 * sampled["mid_ret_100ms"].rolling(window, min_periods=2).std()
        sampled[f"spread_mean_roll_{label}"] = sampled["spread_bps"].rolling(window, min_periods=2).mean()
        sampled[f"spread_std_roll_{label}"] = sampled["spread_bps"].rolling(window, min_periods=2).std()
        sampled[f"microprice_edge_mean_roll_{label}"] = sampled["microprice_edge_bps"].rolling(window, min_periods=2).mean()
        sampled[f"microprice_edge_std_roll_{label}"] = sampled["microprice_edge_bps"].rolling(window, min_periods=2).std()
        sampled[f"microprice_edge_zscore_{label}"] = _safe_div(
            sampled["microprice_edge_bps"] - sampled[f"microprice_edge_mean_roll_{label}"],
            sampled[f"microprice_edge_std_roll_{label}"].replace(0, np.nan),
        )
        sampled[f"buy_pressure_mean_{label}"] = sampled["recent_buy_pressure"].rolling(window, min_periods=1).mean()
        sampled[f"buy_pressure_abs_mean_{label}"] = sampled["recent_buy_pressure"].abs().rolling(window, min_periods=1).mean()
        sampled[f"cancel_pressure_mean_{label}"] = sampled["recent_cancel_pressure"].rolling(window, min_periods=1).mean()
        sampled[f"cancel_to_buy_pressure_{label}"] = _safe_div(
            sampled[f"cancel_pressure_mean_{label}"],
            sampled[f"buy_pressure_abs_mean_{label}"].replace(0, np.nan),
        )
        sampled[f"book_pressure_5_mean_{label}"] = sampled["book_pressure_5"].rolling(window, min_periods=2).mean()
        sampled[f"book_pressure_5_std_{label}"] = sampled["book_pressure_5"].rolling(window, min_periods=2).std()
        sampled[f"book_pressure_5_zscore_{label}"] = _safe_div(
            sampled["book_pressure_5"] - sampled[f"book_pressure_5_mean_{label}"],
            sampled[f"book_pressure_5_std_{label}"].replace(0, np.nan),
        )

    sampled["rv_ratio_1s_3s"] = _safe_div(sampled["rv_1s_bps"], sampled["rv_3s_bps"].replace(0, np.nan))
    sampled["rv_ratio_3s_10s"] = _safe_div(sampled["rv_3s_bps"], sampled["rv_10s_bps"].replace(0, np.nan))
    sampled["spread_rv_ratio_3s"] = _safe_div(sampled["spread_bps"], sampled["rv_3s_bps"].replace(0, np.nan))
    sampled["flow_accel_1s_10s"] = sampled["buy_pressure_mean_1s"] - sampled["buy_pressure_mean_10s"]
    sampled["cancel_accel_1s_10s"] = sampled["cancel_pressure_mean_1s"] - sampled["cancel_pressure_mean_10s"]
    sampled["book_flow_divergence_3s"] = sampled["book_pressure_5_mean_3s"] - sampled["buy_pressure_mean_3s"]
    sampled["depth_pressure_vol_scaled"] = _safe_div(sampled["book_pressure_1"], sampled["rv_3s_bps"].replace(0, np.nan))
    sampled["microprice_pressure_interact"] = sampled["microprice_edge_bps"] * sampled["book_pressure_1"]

    rolling_count_features: dict[str, pd.Series] = {}
    for window_seconds in ROLLING_WINDOWS_SECONDS:
        window = _window_steps(resample_ms, window_seconds)
        label = f"{window_seconds}s"
        rolling_count_features[f"quote_update_count_{label}"] = sampled["quote_price_change_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"spread_change_count_{label}"] = sampled["spread_change_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"bid_up_count_{label}"] = sampled["bid_up_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"bid_down_count_{label}"] = sampled["bid_down_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"ask_up_count_{label}"] = sampled["ask_up_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"ask_down_count_{label}"] = sampled["ask_down_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"buy_trade_count_{label}"] = sampled["buy_trade_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"sell_trade_count_{label}"] = sampled["sell_trade_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"cancel_event_count_{label}"] = sampled["cancel_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"imbalance_flip_count_{label}"] = sampled["imbalance_flip_event"].rolling(window, min_periods=1).sum()
        rolling_count_features[f"cancel_burst_flag_{label}"] = (
            rolling_count_features[f"cancel_event_count_{label}"] >= max(2.0, 0.2 * window)
        ).astype(float)

    elapsed_features = {
        "secs_since_bid_up": _seconds_since_last_event(sampled["bid_up_event"], resample_ms),
        "secs_since_ask_down": _seconds_since_last_event(sampled["ask_down_event"], resample_ms),
        "secs_since_spread_change": _seconds_since_last_event(sampled["spread_change_event"], resample_ms),
        "secs_since_buy_trade": _seconds_since_last_event(sampled["buy_trade_event"], resample_ms),
        "secs_since_sell_trade": _seconds_since_last_event(sampled["sell_trade_event"], resample_ms),
    }
    sampled = pd.concat(
        [
            sampled,
            pd.DataFrame(rolling_count_features, index=sampled.index),
            pd.DataFrame(elapsed_features, index=sampled.index),
        ],
        axis=1,
    )

    # Defragment after many incremental feature inserts to avoid slow column appends below.
    sampled = sampled.copy()

    sampled["cross_imbalance_l1"] = 0.0
    sampled["cross_book_pressure_5"] = 0.0
    sampled["cross_mid_ret_500ms"] = 0.0
    sampled["cross_spread_bps"] = 0.0
    sampled["cross_buy_pressure_mean"] = 0.0
    if other_stock_dfs:
        other_keys = sorted(other_stock_dfs)
        same_group_stock = _same_group_partner(stock, other_keys)
        if same_group_stock is not None:
            other_frame = _coerce_feature_frame(other_stock_dfs[same_group_stock])
            # Cross-stock inputs are aligned on the current timestamp only, so there is no look-ahead bias.
            same_group_cols = other_frame.reindex(sampled.index)[
                ["imbalance_l1", "book_pressure_5", "mid_ret_500ms", "spread_bps"]
            ].fillna(0.0)
            sampled["cross_imbalance_l1"] = same_group_cols["imbalance_l1"].to_numpy(dtype=np.float64)
            sampled["cross_book_pressure_5"] = same_group_cols["book_pressure_5"].to_numpy(dtype=np.float64)
            sampled["cross_mid_ret_500ms"] = same_group_cols["mid_ret_500ms"].to_numpy(dtype=np.float64)
            sampled["cross_spread_bps"] = same_group_cols["spread_bps"].to_numpy(dtype=np.float64)

        cross_group_frames = []
        if stock is not None:
            current_group = STOCK_TO_GROUP.get(stock)
            cross_group_frames = [
                _coerce_feature_frame(other_stock_dfs[other_stock]).reindex(sampled.index)[["buy_pressure_mean_3s"]]
                for other_stock in other_keys
                if STOCK_TO_GROUP.get(other_stock) != current_group
            ]
        if cross_group_frames:
            cross_group_mean = pd.concat(cross_group_frames, axis=1).fillna(0.0).mean(axis=1)
            sampled["cross_buy_pressure_mean"] = cross_group_mean.to_numpy(dtype=np.float64)

    sampled = sampled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return sampled.reset_index().rename(columns={"index": "timestamp"})


def get_feature_columns(use_log_size: bool = True) -> List[str]:
    cols = [
        "spread_bps", "mid_ret_100ms", "mid_ret_500ms", "micro_dev", "microprice_edge_bps",
        "imbalance_l1", "imbalance_l3", "imbalance_l5", "book_pressure_1", "book_pressure_3", "book_pressure_5",
        "depth_sum_bid", "depth_sum_ask", "depth_ratio_5",
        "bid_top_share_1_of_5", "ask_top_share_1_of_5", "bid_top_share_3_of_5", "ask_top_share_3_of_5",
        "depth_concentration_diff_1_5", "depth_concentration_diff_3_5",
        "depth_slope_bid", "depth_slope_ask", "slope_diff_5",
        "recent_buy_pressure", "recent_cancel_pressure", "realized_vol_1s", "time_in_minute",
        "momentum_1s_bps", "momentum_3s_bps", "momentum_10s_bps",
        "mid_zscore_1s", "mid_zscore_3s", "mid_zscore_10s",
        "mid_range_3s_bps", "mid_range_10s_bps",
        "mid_pos_in_range_3s", "mid_pos_in_range_10s",
        "rv_1s_bps", "rv_3s_bps", "rv_10s_bps",
        "spread_mean_roll_1s", "spread_mean_roll_3s", "spread_std_roll_3s",
        "microprice_edge_zscore_1s", "microprice_edge_zscore_3s",
        "buy_pressure_mean_1s", "buy_pressure_mean_3s", "buy_pressure_mean_10s",
        "cancel_pressure_mean_1s", "cancel_pressure_mean_3s", "cancel_pressure_mean_10s",
        "cancel_to_buy_pressure_1s", "cancel_to_buy_pressure_3s", "cancel_to_buy_pressure_10s",
        "book_pressure_5_mean_1s", "book_pressure_5_mean_3s", "book_pressure_5_zscore_3s",
        "rv_ratio_1s_3s", "rv_ratio_3s_10s", "spread_rv_ratio_3s",
        "flow_accel_1s_10s", "cancel_accel_1s_10s", "book_flow_divergence_3s",
        "depth_pressure_vol_scaled", "microprice_pressure_interact",
        "imbalance_pos_flag", "imbalance_neg_flag",
        "quote_update_count_1s", "quote_update_count_3s", "quote_update_count_10s",
        "spread_change_count_1s", "spread_change_count_3s", "spread_change_count_10s",
        "bid_up_count_1s", "bid_up_count_3s", "bid_up_count_10s",
        "bid_down_count_1s", "bid_down_count_3s", "bid_down_count_10s",
        "ask_up_count_1s", "ask_up_count_3s", "ask_up_count_10s",
        "ask_down_count_1s", "ask_down_count_3s", "ask_down_count_10s",
        "buy_trade_count_1s", "buy_trade_count_3s", "buy_trade_count_10s",
        "sell_trade_count_1s", "sell_trade_count_3s", "sell_trade_count_10s",
        "cancel_event_count_1s", "cancel_event_count_3s", "cancel_event_count_10s",
        "imbalance_flip_count_1s", "imbalance_flip_count_3s", "imbalance_flip_count_10s",
        "cancel_burst_flag_1s", "cancel_burst_flag_3s", "cancel_burst_flag_10s",
        "secs_since_bid_up", "secs_since_ask_down", "secs_since_spread_change",
        "secs_since_buy_trade", "secs_since_sell_trade",
        "cross_imbalance_l1", "cross_book_pressure_5", "cross_mid_ret_500ms",
        "cross_spread_bps", "cross_buy_pressure_mean",
    ]
    if use_log_size:
        for i in range(1, BOOK_LEVELS + 1):
            cols += [f"BidSizeLog_{i}", f"AskSizeLog_{i}"]
    return cols


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    if "stock" not in df.columns:
        minutes = df["minute"].drop_duplicates().sort_values().tolist()
        n = len(minutes)
        i1 = int(n * train_frac)
        i2 = int(n * (train_frac + val_frac))
        train_m = set(minutes[:i1])
        val_m = set(minutes[i1:i2])
        test_m = set(minutes[i2:])
        return (
            df[df["minute"].isin(train_m)].reset_index(drop=True),
            df[df["minute"].isin(val_m)].reset_index(drop=True),
            df[df["minute"].isin(test_m)].reset_index(drop=True),
        )

    splits = {"train": [], "val": [], "test": []}
    for _, stock_df in df.groupby("stock", sort=True):
        minutes = stock_df["minute"].drop_duplicates().sort_values().tolist()
        n = len(minutes)
        i1 = int(n * train_frac)
        i2 = int(n * (train_frac + val_frac))
        train_m = set(minutes[:i1])
        val_m = set(minutes[i1:i2])
        test_m = set(minutes[i2:])
        splits["train"].append(stock_df[stock_df["minute"].isin(train_m)])
        splits["val"].append(stock_df[stock_df["minute"].isin(val_m)])
        splits["test"].append(stock_df[stock_df["minute"].isin(test_m)])
    return (
        pd.concat(splits["train"], ignore_index=True),
        pd.concat(splits["val"], ignore_index=True),
        pd.concat(splits["test"], ignore_index=True),
    )


def standardize(train_df: pd.DataFrame, others: List[pd.DataFrame], feature_cols: List[str]):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)
    out = []
    for df in [train_df] + others:
        tmp = df.copy()
        tmp[feature_cols] = (tmp[feature_cols] - mean) / std
        out.append(tmp)
    return out, mean, std


def compute_side_edges(df: pd.DataFrame, horizon_steps: int = 30) -> pd.DataFrame:
    out = df.copy()
    future_mean_ask = pd.Series(index=out.index, dtype=np.float64)
    future_mean_bid = pd.Series(index=out.index, dtype=np.float64)
    future_ask_vol = pd.Series(index=out.index, dtype=np.float64)
    future_bid_vol = pd.Series(index=out.index, dtype=np.float64)
    group_cols = ["minute"] if "stock" not in out.columns else ["stock", "minute"]
    for _, minute_df in out.groupby(group_cols, sort=False):
        future_mean_ask.loc[minute_df.index] = _future_window_stat(minute_df["AskPrice_1"], horizon_steps, stat="mean")
        future_mean_bid.loc[minute_df.index] = _future_window_stat(minute_df["BidPrice_1"], horizon_steps, stat="mean")
        future_ask_vol.loc[minute_df.index] = _future_window_stat(minute_df["AskPrice_1"], horizon_steps, stat="std")
        future_bid_vol.loc[minute_df.index] = _future_window_stat(minute_df["BidPrice_1"], horizon_steps, stat="std")

    out["future_mean_ask"] = future_mean_ask.fillna(out["AskPrice_1"])
    out["future_mean_bid"] = future_mean_bid.fillna(out["BidPrice_1"])
    out["future_ask_vol"] = future_ask_vol.fillna(0.0)
    out["future_bid_vol"] = future_bid_vol.fillna(0.0)
    out["buy_edge"] = (
        (out["future_mean_ask"] - out["AskPrice_1"]) / (out["future_ask_vol"] + 1e-8)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["sell_edge"] = (
        (out["BidPrice_1"] - out["future_mean_bid"]) / (out["future_bid_vol"] + 1e-8)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def fit_side_label_thresholds(
    df: pd.DataFrame,
    horizon_steps: int = 30,
    target_positive_rate: float = 0.50,
) -> Dict[str, Any]:
    target_positive_rate = float(np.clip(target_positive_rate, 0.05, 0.95))
    edged = compute_side_edges(df, horizon_steps=horizon_steps)
    thresholds: Dict[str, Any] = {}
    for side in ("buy", "sell"):
        edge = edged[f"{side}_edge"].replace([np.inf, -np.inf], np.nan).dropna()
        if edge.empty:
            cutoff = 0.0
            positive_rate = 0.0
        else:
            cutoff = float(edge.quantile(1.0 - target_positive_rate))
            positive_rate = float((edge >= cutoff).mean())
        thresholds[side] = {
            "cutoff": cutoff,
            "target_positive_rate": target_positive_rate,
            "actual_positive_rate": positive_rate,
        }
    return thresholds


def make_side_labels(
    df: pd.DataFrame,
    horizon_steps: int = 30,
    threshold_frac_of_spread: float | None = None,
    target_positive_rate: float | None = None,
    cutoffs: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = compute_side_edges(df, horizon_steps=horizon_steps)

    if cutoffs is not None:
        buy_cutoff = float(cutoffs["buy"]["cutoff"] if isinstance(cutoffs["buy"], dict) else cutoffs["buy"])
        sell_cutoff = float(cutoffs["sell"]["cutoff"] if isinstance(cutoffs["sell"], dict) else cutoffs["sell"])
    elif target_positive_rate is not None:
        fitted = fit_side_label_thresholds(
            out,
            horizon_steps=horizon_steps,
            target_positive_rate=target_positive_rate,
        )
        buy_cutoff = float(fitted["buy"]["cutoff"])
        sell_cutoff = float(fitted["sell"]["cutoff"])
    else:
        median_spread = float(out["spread"].replace(0, np.nan).median())
        threshold = max(median_spread * float(threshold_frac_of_spread or 0.10), 1e-6)
        buy_cutoff = -threshold
        sell_cutoff = -threshold

    out["buy_label"] = (out["buy_edge"] >= buy_cutoff).astype(np.int64)
    out["sell_label"] = (out["sell_edge"] >= sell_cutoff).astype(np.int64)
    out["buy_label_cutoff"] = buy_cutoff
    out["sell_label_cutoff"] = sell_cutoff
    return out


def make_direction_labels(df: pd.DataFrame, horizon_steps: int = 30) -> pd.DataFrame:
    out = df.copy()
    future_mid = out["mid"].shift(-horizon_steps)
    delta = future_mid - out["mid"]
    thresh = max(out["spread"].median() * 0.25, 1e-6)
    out["future_delta"] = delta.fillna(0.0)
    out["label"] = 1
    out.loc[delta > thresh, "label"] = 2
    out.loc[delta < -thresh, "label"] = 0
    return out


def feature_horizon_correlation_table(
    df: pd.DataFrame,
    feature_cols: List[str],
    base_horizon_steps: int,
    side: str | None = None,
) -> pd.DataFrame:
    horizons = {
        "corr_ret_0p5x_horizon": max(int(round(base_horizon_steps * 0.5)), 1),
        "corr_ret_1p0x_horizon": max(int(round(base_horizon_steps * 1.0)), 1),
        "corr_ret_2p0x_horizon": max(int(round(base_horizon_steps * 2.0)), 1),
    }
    target_sides = [side] if side is not None else ["buy", "sell"]
    target_by_side: Dict[str, Dict[str, pd.Series]] = {target_side: {} for target_side in target_sides}
    for label, steps in horizons.items():
        edged = compute_side_edges(df, horizon_steps=steps)
        for target_side in target_sides:
            target_by_side[target_side][label] = edged[f"{target_side}_edge"].replace([np.inf, -np.inf], np.nan)

    rows = []
    for target_side in target_sides:
        for feature in feature_cols:
            row: Dict[str, Any] = {"feature": feature, "side": target_side}
            x = df[feature].replace([np.inf, -np.inf], np.nan)
            for label, y in target_by_side[target_side].items():
                pair = pd.DataFrame({"x": x, "y": y}).dropna()
                row[label] = float(pair["x"].corr(pair["y"], method="spearman")) if len(pair) >= 2 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def minute_groups(df: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
    group_cols: List[str] = ["minute"]
    if "stock" in df.columns:
        group_cols = ["stock", "minute"]
    return {m: g.reset_index(drop=True) for m, g in df.groupby(group_cols, sort=True)}


def benchmark_price(minute_df: pd.DataFrame, side: str) -> float:
    row = minute_df.iloc[0]
    return float(row["AskPrice_1"] if side == "buy" else row["BidPrice_1"])


def theoretical_best_price(minute_df: pd.DataFrame, side: str) -> float:
    if side == "buy":
        return float(minute_df["AskPrice_1"].min())
    return float(minute_df["BidPrice_1"].max())


def execute_price(row: pd.Series, side: str) -> float:
    return float(row["AskPrice_1"] if side == "buy" else row["BidPrice_1"])


def trade_improvement(exec_price: float, benchmark: float, side: str) -> float:
    return benchmark - exec_price if side == "buy" else exec_price - benchmark


def theoretical_best_improvement(minute_df: pd.DataFrame, side: str) -> float:
    return trade_improvement(theoretical_best_price(minute_df, side), benchmark_price(minute_df, side), side)

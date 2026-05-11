from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import INPUT_DIR, KNOWN_STOCKS
from .data_utils import build_features, get_feature_columns, load_lob_csv


def model_key(stock: str, side: str) -> str:
    return f"{stock}_{side}"


def split_train_val_test(
    feat_df: pd.DataFrame,
    train_frac: float = 0.64,
    val_frac: float = 0.16,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minutes = feat_df["minute"].drop_duplicates().sort_values().tolist()
    i1 = int(len(minutes) * train_frac)
    i2 = int(len(minutes) * (train_frac + val_frac))
    train_minutes = set(minutes[:i1])
    val_minutes = set(minutes[i1:i2])
    test_minutes = set(minutes[i2:])
    return (
        feat_df[feat_df["minute"].isin(train_minutes)].reset_index(drop=True),
        feat_df[feat_df["minute"].isin(val_minutes)].reset_index(drop=True),
        feat_df[feat_df["minute"].isin(test_minutes)].reset_index(drop=True),
    )


def split_train_val_only(
    feat_df: pd.DataFrame,
    train_frac_within_history: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    minutes = feat_df["minute"].drop_duplicates().sort_values().tolist()
    split_idx = max(1, min(int(len(minutes) * train_frac_within_history), len(minutes) - 1))
    train_minutes = set(minutes[:split_idx])
    val_minutes = set(minutes[split_idx:])
    return (
        feat_df[feat_df["minute"].isin(train_minutes)].reset_index(drop=True),
        feat_df[feat_df["minute"].isin(val_minutes)].reset_index(drop=True),
    )


def build_feature_frames_from_paths(cfg: dict, file_map: dict[str, Path]) -> dict[str, pd.DataFrame]:
    raw_by_stock: dict[str, pd.DataFrame] = {}
    base_by_stock: dict[str, pd.DataFrame] = {}
    for stock, path in file_map.items():
        raw = load_lob_csv(path)
        raw["stock"] = stock
        raw_by_stock[stock] = raw
        base_by_stock[stock] = build_features(
            raw,
            resample_ms=cfg["feature"]["resample_ms"],
            use_log_size=cfg["feature"]["use_log_size"],
            stock=stock,
        )

    frames: dict[str, pd.DataFrame] = {}
    available = list(file_map)
    for stock in available:
        other = {other_stock: base_by_stock[other_stock] for other_stock in available if other_stock != stock}
        feat_df = build_features(
            raw_by_stock[stock],
            resample_ms=cfg["feature"]["resample_ms"],
            use_log_size=cfg["feature"]["use_log_size"],
            stock=stock,
            other_stock_dfs=other if other else None,
        )
        feat_df["time_in_minute_raw"] = feat_df["time_in_minute"]
        frames[stock] = feat_df
    return frames


def build_feature_frames(cfg: dict, include_known_only: bool = True) -> dict[str, pd.DataFrame]:
    train_map = {}
    configured = cfg["files"]["train_by_stock"]
    stocks = KNOWN_STOCKS if include_known_only else tuple(configured)
    for stock in stocks:
        filename = configured.get(stock)
        if filename is None:
            continue
        train_map[stock] = INPUT_DIR / filename
    return build_feature_frames_from_paths(cfg, train_map)


def feature_columns(use_log_size: bool = True) -> list[str]:
    return get_feature_columns(use_log_size=use_log_size)


def benchmark_price(minute_df: pd.DataFrame, side: str) -> float:
    row = minute_df.iloc[0]
    return float(row["AskPrice_1"] if side == "buy" else row["BidPrice_1"])


def theoretical_best_price(minute_df: pd.DataFrame, side: str) -> float:
    if side == "buy":
        return float(minute_df["AskPrice_1"].min())
    return float(minute_df["BidPrice_1"].max())


def improvement(exec_price: float, benchmark: float, side: str) -> float:
    return float(benchmark - exec_price) if side == "buy" else float(exec_price - benchmark)


def summarize_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        summary = (
            df.groupby(group_cols, as_index=False)
            .agg(
                avg_exec_price=("exec_price", "mean"),
                avg_benchmark=("benchmark_price", "mean"),
                avg_improvement=("improvement", "mean"),
                median_improvement=("improvement", "median"),
                std_improvement=("improvement", "std"),
                theoretical_best_avg_improvement=("theoretical_best_improvement", "mean"),
                avg_exec_sec_into_minute=("sec_into_minute", "mean"),
                win_rate=("improvement", lambda x: float((x > 0).mean())),
                n_trades=("improvement", "size"),
            )
        )
        summary["std_improvement"] = summary["std_improvement"].fillna(0.0)
        denom = summary["theoretical_best_avg_improvement"].replace(0.0, np.nan)
        summary["model_pct_of_theoretical_best"] = (
            summary["avg_improvement"] / denom
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return summary

    stock_side = _aggregate(trades_df, ["stock", "side"])
    side_overall = _aggregate(trades_df, ["side"])
    side_overall.insert(0, "stock", "ALL")
    overall = _aggregate(trades_df.assign(side="ALL"), ["side"]).rename(columns={"side": "stock"})
    overall.insert(1, "side", "ALL")
    overall["stock"] = "ALL"
    return pd.concat([stock_side, side_overall, overall], ignore_index=True, sort=False).sort_values(
        ["stock", "side"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_cost_improvement(trades_df: pd.DataFrame, split_col: str = "dataset_split") -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_split",
                "stock",
                "n_buy_trades",
                "n_sell_trades",
                "total_youralgo_buy",
                "total_youralgo_sell",
                "total_twap_buy",
                "total_twap_sell",
                "youralgo_cost",
                "twap_cost",
                "pct_improvement_vs_twap",
            ]
        )
    if split_col not in trades_df.columns:
        raise ValueError(f"Expected split column '{split_col}' in trades_df.")

    def _aggregate(df: pd.DataFrame, dataset_split: str, stock: str) -> dict[str, Any]:
        buy_df = df[df["side"] == "buy"]
        sell_df = df[df["side"] == "sell"]
        total_youralgo_buy = float(buy_df["exec_price"].sum())
        total_youralgo_sell = float(sell_df["exec_price"].sum())
        total_twap_buy = float(buy_df["benchmark_price"].sum())
        total_twap_sell = float(sell_df["benchmark_price"].sum())
        youralgo_cost = total_youralgo_buy - total_youralgo_sell
        twap_cost = total_twap_buy - total_twap_sell
        pct_improvement = 100.0 - 100.0 * (youralgo_cost / twap_cost) if abs(twap_cost) > 1e-12 else np.nan
        return {
            "dataset_split": dataset_split,
            "stock": stock,
            "n_buy_trades": int(len(buy_df)),
            "n_sell_trades": int(len(sell_df)),
            "total_youralgo_buy": total_youralgo_buy,
            "total_youralgo_sell": total_youralgo_sell,
            "total_twap_buy": total_twap_buy,
            "total_twap_sell": total_twap_sell,
            "youralgo_cost": youralgo_cost,
            "twap_cost": twap_cost,
            "pct_improvement_vs_twap": float(pct_improvement) if pd.notna(pct_improvement) else np.nan,
        }

    rows: list[dict[str, Any]] = []
    for dataset_split, split_df in trades_df.groupby(split_col, sort=True):
        for stock, stock_df in split_df.groupby("stock", sort=True):
            rows.append(_aggregate(stock_df, dataset_split=str(dataset_split), stock=str(stock)))
        rows.append(_aggregate(split_df, dataset_split=str(dataset_split), stock="ALL"))
    return pd.DataFrame(rows).sort_values(["dataset_split", "stock"], kind="stable").reset_index(drop=True)

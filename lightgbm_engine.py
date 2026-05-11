from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import benchmark_price, improvement, model_key, theoretical_best_price
from .panel import ridge_minute_return_series
from .score_logic import (
    apply_signal_ewm,
    compile_gate_rules,
    compile_rules,
    fit_rule_thresholds,
    resolve_signal_ewm_span,
    rule_flag,
    score_seconds,
)

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
except ImportError as exc:  # pragma: no cover
    raise ImportError("lightgbm is required for execution_algorithm.") from exc


@dataclass
class LightGBMRegressorModel:
    features: list[str]
    params: dict[str, Any]
    seed: int
    constant_prediction: float | None = None
    estimator: LGBMRegressor | None = None
    best_iteration_: int | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        val_df: pd.DataFrame | None = None,
        use_early_stopping: bool = True,
    ) -> None:
        x_train = train_df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_train = train_df[target_col].astype(float)
        if y_train.nunique(dropna=False) <= 1:
            self.constant_prediction = float(y_train.iloc[0]) if len(y_train) else 0.0
            self.estimator = None
            self.best_iteration_ = 1
            return

        callbacks = [log_evaluation(period=0)]
        fit_kwargs: dict[str, Any] = {}
        if (
            use_early_stopping
            and val_df is not None
            and not val_df.empty
            and val_df[target_col].astype(float).nunique(dropna=False) > 1
        ):
            x_val = val_df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_val = val_df[target_col].astype(float)
            fit_kwargs["eval_set"] = [(x_val, y_val)]
            fit_kwargs["eval_metric"] = "l2"
            callbacks.append(early_stopping(stopping_rounds=40, verbose=False))

        self.constant_prediction = None
        self.estimator = LGBMRegressor(
            objective="regression",
            random_state=int(self.seed),
            n_jobs=1,
            verbosity=-1,
            **self.params,
        )
        self.estimator.fit(x_train, y_train, callbacks=callbacks, **fit_kwargs)
        best_iteration = getattr(self.estimator, "best_iteration_", None)
        if best_iteration is None or int(best_iteration) <= 0:
            best_iteration = int(self.params["n_estimators"])
        self.best_iteration_ = int(best_iteration)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.constant_prediction is not None:
            return np.full(len(x), self.constant_prediction, dtype=float)
        assert self.estimator is not None
        num_iteration = self.best_iteration_ if self.best_iteration_ is not None else None
        return self.estimator.predict(x, num_iteration=num_iteration)

    def feature_importance(self) -> pd.DataFrame:
        if self.estimator is None:
            return pd.DataFrame(
                {
                    "feature": self.features,
                    "split_importance": np.zeros(len(self.features), dtype=float),
                    "gain_importance": np.zeros(len(self.features), dtype=float),
                }
            )
        booster = self.estimator.booster_
        return pd.DataFrame(
            {
                "feature": self.features,
                "split_importance": booster.feature_importance(importance_type="split"),
                "gain_importance": booster.feature_importance(importance_type="gain"),
            }
        ).sort_values(["gain_importance", "split_importance"], ascending=[False, False], kind="stable")


def resolve_feature_set(cfg: dict, available_features: list[str], stock: str, side: str) -> list[str]:
    ordered: list[str] = []

    def _append(features: list[str]) -> None:
        for feature in features:
            if feature in available_features and feature not in ordered:
                ordered.append(feature)

    feature_override = cfg["lightgbm"].get("feature_overrides_by_model", {}).get(model_key(stock, side), [])
    if feature_override:
        return [feature for feature in feature_override if feature in available_features]

    score_rules = cfg["score"]["rules_by_model"].get(model_key(stock, side), [])
    gate_rules = cfg["score"].get("gate_rules_by_model", {}).get(model_key(stock, side), [])
    _append([rule["feature"] for rule in score_rules])
    _append([rule["feature"] for rule in gate_rules])
    _append(cfg["lightgbm"].get("feature_extras_by_model", {}).get(model_key(stock, side), []))
    _append(cfg["lightgbm"].get("global_safe_features", []))
    max_features = int(cfg["lightgbm"].get("max_features_per_model", len(ordered)))
    return ordered[:max_features]


def uses_score_logic_feature(cfg: dict, stock: str, side: str) -> bool:
    return not bool(cfg["lightgbm"].get("disable_score_logic_by_model", {}).get(model_key(stock, side), False))


def fit_score_logic_state(panel: pd.DataFrame, calibration_minutes: list[pd.Timestamp], cfg: dict, rule_stock: str, side: str) -> dict:
    train_panel = panel[panel["minute"].isin(calibration_minutes)].copy()
    score_rules = fit_rule_thresholds(train_panel, compile_rules(cfg, rule_stock, side))
    gate_rules = fit_rule_thresholds(train_panel, compile_gate_rules(cfg, rule_stock, side))
    span = resolve_signal_ewm_span(cfg, rule_stock, side)
    return {"score_rules": score_rules, "score_gate_rules": gate_rules, "signal_ewm_span": span}


def apply_score_logic_state(panel: pd.DataFrame, score_state: dict) -> pd.DataFrame:
    scored = score_seconds(panel, score_state["score_rules"], compiled_gate_rules=score_state["score_gate_rules"])
    scored = apply_signal_ewm(scored, span=score_state["signal_ewm_span"])
    feature_df = scored[["minute", "second_in_minute", "timestamp", "signal_score"]].rename(
        columns={"signal_score": "score_logic_signal"}
    )
    merged = panel.merge(feature_df, on=["minute", "second_in_minute", "timestamp"], how="left", sort=False)
    merged["score_logic_signal"] = merged["score_logic_signal"].fillna(0.0)
    return merged


def fit_execution_gate_rules(train_panel: pd.DataFrame, cfg: dict, rule_stock: str, side: str) -> list[dict]:
    return fit_rule_thresholds(train_panel, compile_gate_rules(cfg, rule_stock, side))


def apply_gate_rules(panel: pd.DataFrame, gate_rules: list[dict]) -> pd.DataFrame:
    gated = panel.copy()
    gated["gate_pass"] = 1.0
    if not gate_rules:
        return gated
    gate_cols: list[str] = []
    for idx, rule in enumerate(gate_rules, start=1):
        col = f"gate_{idx}_{rule['feature']}"
        gated[col] = rule_flag(gated, rule)
        gate_cols.append(col)
    gated["gate_pass"] = gated[gate_cols].min(axis=1) if gate_cols else 1.0
    return gated


def fit_threshold(train_scores: np.ndarray, threshold_quantile: float) -> float:
    if train_scores.size == 0:
        return 0.0
    q = float(np.clip(threshold_quantile, 0.01, 0.99))
    return float(np.quantile(train_scores, 1.0 - q))


def select_execution_row(minute_df: pd.DataFrame, threshold: float) -> tuple[pd.Series, dict]:
    scored = minute_df.sort_values("second_in_minute").copy()
    hits = scored[(scored["gate_pass"] >= 1.0) & (scored["signal_score"] >= threshold)]
    if not hits.empty:
        picked = hits.iloc[0]
        threshold_hit = True
        forced_last_step = False
    else:
        picked = scored.iloc[-1]
        threshold_hit = False
        forced_last_step = True
    return picked, {"threshold_hit": threshold_hit, "forced_last_step": forced_last_step}


def evaluate_threshold(
    scored_panel: pd.DataFrame,
    raw_feat_df: pd.DataFrame,
    stock: str,
    side: str,
    minutes: list[pd.Timestamp],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_feat = raw_feat_df.copy()
    raw_feat["minute"] = pd.to_datetime(raw_feat["minute"])
    trades: list[dict[str, Any]] = []
    signal_rows: list[pd.DataFrame] = []
    for minute in minutes:
        minute_df = scored_panel[scored_panel["minute"] == minute].copy()
        if minute_df.empty:
            continue
        minute_df["threshold"] = float(threshold)
        picked, meta = select_execution_row(minute_df, threshold=float(threshold))
        minute_df["selected_exec"] = (minute_df["timestamp"] == picked["timestamp"]).astype(int)
        signal_rows.append(minute_df.copy())

        minute_raw = raw_feat[raw_feat["minute"] == minute].reset_index(drop=True)
        exec_price = float(picked["AskPrice_1"] if side == "buy" else picked["BidPrice_1"])
        benchmark = benchmark_price(minute_raw, side)
        theoretical = theoretical_best_price(minute_raw, side)
        trades.append(
            {
                "stock": stock,
                "side": side,
                "minute": pd.Timestamp(minute),
                "exec_time": pd.Timestamp(picked["timestamp"]),
                "exec_price": exec_price,
                "benchmark_price": benchmark,
                "theoretical_best_price": theoretical,
                "theoretical_best_improvement": improvement(theoretical, benchmark, side),
                "improvement": improvement(exec_price, benchmark, side),
                "sec_into_minute": float(picked["time_in_minute_raw"] * 60.0),
                "signal_score": float(picked["signal_score"]),
                "signal_threshold": float(threshold),
                "threshold_hit": bool(meta["threshold_hit"]),
                "forced_last_step": bool(meta["forced_last_step"]),
            }
        )
    return pd.DataFrame(trades), pd.concat(signal_rows, ignore_index=True) if signal_rows else pd.DataFrame()


def score_trade_objective(trades_df: pd.DataFrame, cfg: dict) -> float:
    if trades_df.empty:
        return -1e9
    mean_improvement = float(trades_df["improvement"].mean())
    stability = float(trades_df["improvement"].std(ddof=0)) if len(trades_df) > 1 else 0.0
    forced_last_rate = float(trades_df["forced_last_step"].mean())
    return (
        mean_improvement
        - float(cfg["lightgbm"]["forced_last_penalty"]) * forced_last_rate
        - float(cfg["lightgbm"]["stability_penalty"]) * stability
    )


def select_model_and_threshold(
    train_panel: pd.DataFrame,
    val_panel: pd.DataFrame,
    raw_feat_df: pd.DataFrame,
    stock: str,
    side: str,
    features: list[str],
    cfg: dict,
    logger,
    rule_stock: str,
) -> dict:
    gate_rules = fit_execution_gate_rules(train_panel, cfg, rule_stock, side)
    train_panel = apply_gate_rules(train_panel, gate_rules)
    val_panel = apply_gate_rules(val_panel, gate_rules)
    target_col = "target_return_bps"
    val_minutes = sorted(pd.to_datetime(val_panel["minute"]).drop_duplicates().tolist())
    grid_rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for param_set in cfg["lightgbm"]["candidate_param_sets"]:
        params = {k: v for k, v in param_set.items() if k != "name"}
        model = LightGBMRegressorModel(features=list(features), params=params, seed=int(cfg["seed"]))
        model.fit(train_panel, target_col=target_col, val_df=val_panel, use_early_stopping=True)
        train_scores = model.predict(train_panel)
        val_scored = val_panel.copy()
        val_scored["signal_score"] = model.predict(val_panel)
        for threshold_quantile in cfg["lightgbm"]["threshold_quantile_grid"]:
            threshold = fit_threshold(train_scores, threshold_quantile=float(threshold_quantile))
            trades_df, _ = evaluate_threshold(
                val_scored,
                raw_feat_df=raw_feat_df,
                stock=stock,
                side=side,
                minutes=val_minutes,
                threshold=threshold,
            )
            objective = score_trade_objective(trades_df, cfg)
            row = {
                "stock": stock,
                "side": side,
                "rule_template_stock": rule_stock,
                "param_set": param_set["name"],
                "threshold_quantile": float(threshold_quantile),
                "threshold": float(threshold),
                "objective": float(objective),
                "val_avg_improvement": float(trades_df["improvement"].mean()) if not trades_df.empty else np.nan,
                "val_win_rate": float((trades_df["improvement"] > 0).mean()) if not trades_df.empty else np.nan,
                "val_forced_last_rate": float(trades_df["forced_last_step"].mean()) if not trades_df.empty else np.nan,
                "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
            }
            grid_rows.append(row)
            if best_result is None or objective > best_result["objective"]:
                best_result = {
                    "param_set_name": param_set["name"],
                    "params": dict(params),
                    "threshold_quantile": float(threshold_quantile),
                    "threshold": float(threshold),
                    "objective": float(objective),
                    "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
                    "gate_rules": gate_rules,
                }
        logger.info(
            "LGBM[%s][%s] evaluated param_set=%s best_iteration=%d",
            stock,
            side,
            param_set["name"],
            int(model.best_iteration_ or params["n_estimators"]),
        )

    assert best_result is not None
    best_result["validation_grid"] = pd.DataFrame(grid_rows).sort_values(
        ["objective", "val_avg_improvement"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    return best_result


def fit_lightgbm_artifact(
    full_panel: pd.DataFrame,
    raw_feat_df: pd.DataFrame,
    stock: str,
    side: str,
    selected_features: list[str],
    train_minutes: list[pd.Timestamp],
    val_minutes: list[pd.Timestamp],
    cfg: dict,
    logger,
    rule_stock: str | None = None,
    use_score_logic: bool | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fit the execution model and return a serializable artifact plus validation rows."""
    rule_stock = rule_stock or stock
    use_score_logic = uses_score_logic_feature(cfg, rule_stock, side) if use_score_logic is None else bool(use_score_logic)

    selection_score_state = fit_score_logic_state(full_panel, train_minutes, cfg, rule_stock, side)
    selection_panel = apply_score_logic_state(full_panel, selection_score_state)
    model_features = list(selected_features)
    if use_score_logic and "score_logic_signal" not in model_features:
        model_features.append("score_logic_signal")

    train_panel = selection_panel[selection_panel["minute"].isin(train_minutes)].copy()
    val_panel = selection_panel[selection_panel["minute"].isin(val_minutes)].copy()
    if train_panel.empty or val_panel.empty:
        raise ValueError(f"Empty train/validation split for {stock} {side}.")

    selection = select_model_and_threshold(
        train_panel=train_panel,
        val_panel=val_panel,
        raw_feat_df=raw_feat_df,
        stock=stock,
        side=side,
        features=model_features,
        cfg=cfg,
        logger=logger,
        rule_stock=rule_stock,
    )

    fit_minutes = sorted(set(train_minutes).union(val_minutes)) if cfg["lightgbm"].get("refit_on_train_plus_val", True) else list(train_minutes)
    final_score_state = fit_score_logic_state(full_panel, fit_minutes, cfg, rule_stock, side)
    final_panel = apply_score_logic_state(full_panel, final_score_state)
    fit_panel = final_panel[final_panel["minute"].isin(fit_minutes)].copy()
    final_gate_rules = fit_execution_gate_rules(fit_panel, cfg, rule_stock, side)
    fit_with_gates = apply_gate_rules(fit_panel, final_gate_rules)

    final_model = LightGBMRegressorModel(features=list(model_features), params=dict(selection["params"]), seed=int(cfg["seed"]))
    final_model.params["n_estimators"] = int(selection["best_iteration"])
    final_model.fit(fit_with_gates, target_col="target_return_bps", val_df=None, use_early_stopping=False)
    fit_scores = final_model.predict(fit_with_gates)
    final_threshold = fit_threshold(fit_scores, threshold_quantile=float(selection["threshold_quantile"]))

    artifact = {
        "stock": stock,
        "side": side,
        "rule_stock": rule_stock,
        "selected_features": list(selected_features),
        "model_features": list(model_features),
        "use_score_logic": bool(use_score_logic),
        "score_state": final_score_state,
        "execution_gate_rules": final_gate_rules,
        "threshold_quantile": float(selection["threshold_quantile"]),
        "threshold": float(final_threshold),
        "param_set": str(selection["param_set_name"]),
        "best_iteration": int(selection["best_iteration"]),
        "objective": float(selection["objective"]),
        "params": dict(final_model.params),
        "model": final_model,
        "feature_importance": final_model.feature_importance(),
        "fit_minutes": [pd.Timestamp(x) for x in fit_minutes],
        "validation_grid": selection["validation_grid"],
    }
    return artifact, selection["validation_grid"]


def apply_lightgbm_artifact(
    full_panel: pd.DataFrame,
    raw_feat_df: pd.DataFrame,
    artifact: dict,
    stock: str,
    side: str,
    minutes: list[pd.Timestamp],
    dataset_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_panel = apply_score_logic_state(full_panel, artifact["score_state"])
    scored_panel = apply_gate_rules(scored_panel, artifact["execution_gate_rules"])
    scored_panel["signal_score"] = artifact["model"].predict(scored_panel)
    scored_panel["target_return_bps"] = ridge_minute_return_series(scored_panel, side)
    scored_panel["stock"] = stock
    scored_panel["side"] = side
    scored_panel["signal_threshold"] = float(artifact["threshold"])
    trades_df, signals_df = evaluate_threshold(
        scored_panel,
        raw_feat_df=raw_feat_df,
        stock=stock,
        side=side,
        minutes=minutes,
        threshold=float(artifact["threshold"]),
    )
    if not trades_df.empty:
        trades_df["dataset_split"] = dataset_split
        trades_df["threshold_quantile"] = float(artifact["threshold_quantile"])
        trades_df["param_set"] = str(artifact["param_set"])
        trades_df["best_iteration"] = int(artifact["best_iteration"])
        trades_df["strategy"] = "execution_algorithm_lightgbm"
    if not signals_df.empty:
        signals_df["stock"] = stock
        signals_df["side"] = side
        signals_df["signal_threshold"] = float(artifact["threshold"])
    return trades_df, signals_df

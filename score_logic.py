from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .panel import model_key


@dataclass
class ScoreRule:
    feature: str
    op: str
    quantile: float | None = None
    weight: float = 1.0
    lower_quantile: float | None = None
    upper_quantile: float | None = None


def compile_rules(cfg: dict, stock: str, side: str) -> list[ScoreRule]:
    raw_rules = cfg["score"]["rules_by_model"].get(model_key(stock, side), [])
    return [ScoreRule(**rule) for rule in raw_rules]


def compile_gate_rules(cfg: dict, stock: str, side: str) -> list[dict]:
    return list(cfg["score"].get("gate_rules_by_model", {}).get(model_key(stock, side), []))


def resolve_signal_ewm_span(cfg: dict, stock: str, side: str) -> int | None:
    span = cfg["score"].get("signal_ewm_span_by_model", {}).get(model_key(stock, side))
    if span is None:
        return None
    span_value = int(span)
    return span_value if span_value > 1 else None


def fit_rule_thresholds(train_panel: pd.DataFrame, rules: list[dict | ScoreRule]) -> list[dict]:
    compiled: list[dict] = []
    for rule in rules:
        feature = rule["feature"] if isinstance(rule, dict) else rule.feature
        op = rule["op"] if isinstance(rule, dict) else rule.op
        raw_quantile = rule.get("quantile") if isinstance(rule, dict) else rule.quantile
        quantile = float(raw_quantile) if raw_quantile is not None else None
        lower_quantile_raw = rule.get("lower_quantile") if isinstance(rule, dict) else rule.lower_quantile
        upper_quantile_raw = rule.get("upper_quantile") if isinstance(rule, dict) else rule.upper_quantile
        lower_quantile = float(lower_quantile_raw) if lower_quantile_raw is not None else None
        upper_quantile = float(upper_quantile_raw) if upper_quantile_raw is not None else None
        weight = float(rule.get("weight", 0.0) if isinstance(rule, dict) else rule.weight)
        if feature not in train_panel.columns:
            continue
        series = train_panel[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        if op in {"ge", "le"}:
            if quantile is None:
                raise ValueError(f"Rule {feature} with op={op} requires quantile.")
            compiled.append(
                {
                    "feature": feature,
                    "op": op,
                    "quantile": quantile,
                    "threshold_value": float(series.quantile(quantile)),
                    "weight": weight,
                }
            )
        elif op in {"between", "outside"}:
            if lower_quantile is None or upper_quantile is None:
                raise ValueError(f"Rule {feature} with op={op} requires lower_quantile and upper_quantile.")
            lower_value = float(series.quantile(lower_quantile))
            upper_value = float(series.quantile(upper_quantile))
            if lower_value > upper_value:
                lower_value, upper_value = upper_value, lower_value
                lower_quantile, upper_quantile = upper_quantile, lower_quantile
            compiled.append(
                {
                    "feature": feature,
                    "op": op,
                    "lower_quantile": lower_quantile,
                    "upper_quantile": upper_quantile,
                    "lower_threshold_value": lower_value,
                    "upper_threshold_value": upper_value,
                    "weight": weight,
                }
            )
        else:
            raise ValueError(f"Unsupported rule op {op}")
    return compiled


def rule_flag(scored: pd.DataFrame, rule: dict) -> pd.Series:
    feature = rule["feature"]
    op = rule["op"]
    if op == "ge":
        return (scored[feature] >= float(rule["threshold_value"])).astype(float)
    if op == "le":
        return (scored[feature] <= float(rule["threshold_value"])).astype(float)
    if op == "between":
        return (
            (scored[feature] >= float(rule["lower_threshold_value"]))
            & (scored[feature] <= float(rule["upper_threshold_value"]))
        ).astype(float)
    if op == "outside":
        return (
            (scored[feature] <= float(rule["lower_threshold_value"]))
            | (scored[feature] >= float(rule["upper_threshold_value"]))
        ).astype(float)
    raise ValueError(f"Unsupported rule op {op}")


def score_seconds(panel: pd.DataFrame, compiled_rules: list[dict], compiled_gate_rules: list[dict] | None = None) -> pd.DataFrame:
    scored = panel.copy()
    scored["signal_score"] = 0.0
    scored["gate_pass"] = 1.0
    component_cols: list[str] = []
    for idx, rule in enumerate(compiled_rules, start=1):
        feature = rule["feature"]
        weight = float(rule["weight"])
        col = f"rule_{idx}_{feature}"
        scored[col] = weight * rule_flag(scored, rule)
        component_cols.append(col)
    if component_cols:
        scored["signal_score"] = scored[component_cols].sum(axis=1)
    if compiled_gate_rules:
        gate_cols: list[str] = []
        for idx, rule in enumerate(compiled_gate_rules, start=1):
            feature = rule["feature"]
            col = f"gate_{idx}_{feature}"
            scored[col] = rule_flag(scored, rule)
            gate_cols.append(col)
        scored["gate_pass"] = scored[gate_cols].min(axis=1) if gate_cols else 1.0
    return scored


def apply_signal_ewm(scored_panel: pd.DataFrame, span: int | None) -> pd.DataFrame:
    scored = scored_panel.sort_values(["minute", "second_in_minute"], kind="stable").copy()
    scored["raw_signal_score"] = scored["signal_score"]
    if span is None:
        scored["signal_ewm_span"] = np.nan
        return scored
    group_cols = ["minute"] if "stock" not in scored.columns else ["stock", "minute"]
    scored["signal_score"] = scored.groupby(group_cols, sort=False)["signal_score"].transform(
        lambda s: s.ewm(span=span, adjust=False).mean()
    )
    scored["signal_ewm_span"] = float(span)
    return scored

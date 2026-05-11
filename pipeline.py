from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import INPUT_DIR, KNOWN_STOCKS, OUTPUT_DIR, ensure_dirs, load_pickle, make_logger, progress_bar, save_pickle, set_seed, stock_model_path
from .config import execution_config
from .data import (
    build_feature_frames,
    build_feature_frames_from_paths,
    feature_columns,
    model_key,
    split_train_val_only,
    summarize_cost_improvement,
    summarize_trades,
)
from .features import augment_feature_frame, fit_augmentation_state
from .lightgbm_engine import (
    apply_lightgbm_artifact,
    fit_lightgbm_artifact,
    resolve_feature_set,
    uses_score_logic_feature,
)
from .panel import build_second_panel, ridge_minute_return_series


def _available_features(cfg: dict) -> list[str]:
    return feature_columns(use_log_size=cfg["feature"]["use_log_size"]) + cfg["lightgbm"].get("extra_feature_columns", [])


def _artifact_model_row(artifact: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock": artifact["stock"],
                "side": artifact["side"],
                "rule_template_stock": artifact["rule_stock"],
                "param_set": artifact["param_set"],
                "threshold_quantile": float(artifact["threshold_quantile"]),
                "threshold": float(artifact["threshold"]),
                "best_iteration": int(artifact["best_iteration"]),
                "objective": float(artifact["objective"]),
                "n_features": int(len(artifact["model_features"])),
                "features": ",".join(artifact["model_features"]),
                "use_score_logic": bool(artifact["use_score_logic"]),
            }
        ]
    )


def _prepare_augmented_stock_frame(base_df: pd.DataFrame, train_reference: pd.DataFrame, selected_features: list[str], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    augmentation_state = fit_augmentation_state(train_reference)
    augmented = augment_feature_frame(base_df, augmentation_state)
    second_panel = build_second_panel(
        augmented,
        selected_features=selected_features,
        top_decile=float(cfg["lightgbm"]["top_decile"]),
    )
    return augmented, second_panel, augmentation_state


def _write_bundle(output_dir: Path, trades_df: pd.DataFrame, signals_df: pd.DataFrame, feature_importance_df: pd.DataFrame, validation_df: pd.DataFrame, extras: dict[str, pd.DataFrame] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    summarize_trades(trades_df).to_csv(output_dir / "summary.csv", index=False)
    summarize_cost_improvement(trades_df).to_csv(output_dir / "cost_improvement_metrics.csv", index=False)
    signals_df.to_csv(output_dir / "signals.csv", index=False)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    validation_df.to_csv(output_dir / "validation_selection.csv", index=False)
    for name, frame in (extras or {}).items():
        frame.to_csv(output_dir / name, index=False)


def train_known_stock_models(cfg: dict | None = None) -> Path:
    """Train the current best LightGBM execution system on all known-stock train data."""
    cfg = cfg or execution_config()
    ensure_dirs()
    set_seed(int(cfg["seed"]))
    output_dir = OUTPUT_DIR / "train"
    logger = make_logger("execution_algorithm_train", output_dir / "run.log")
    available_features = _available_features(cfg)
    feature_frames = build_feature_frames(cfg)

    trades_all: list[pd.DataFrame] = []
    signals_all: list[pd.DataFrame] = []
    feature_importances_all: list[pd.DataFrame] = []
    validation_rows_all: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    total = len(KNOWN_STOCKS) * 2
    step = 0
    for stock in KNOWN_STOCKS:
        base_df = feature_frames[stock].copy()
        all_minutes = sorted(pd.to_datetime(base_df["minute"]).drop_duplicates().tolist())
        for side in ("buy", "sell"):
            step += 1
            print(f"{progress_bar(step, total)} EXECUTION_ALGO train {stock} {side}", flush=True)
            selected_features = resolve_feature_set(cfg, available_features, stock, side)
            augmented, second_panel, augmentation_state = _prepare_augmented_stock_frame(
                base_df=base_df,
                train_reference=base_df,
                selected_features=selected_features,
                cfg=cfg,
            )
            second_panel["target_return_bps"] = ridge_minute_return_series(second_panel, side)
            artifact, validation_grid = fit_lightgbm_artifact(
                full_panel=second_panel,
                raw_feat_df=augmented,
                stock=stock,
                side=side,
                selected_features=selected_features,
                train_minutes=all_minutes,
                val_minutes=all_minutes,
                cfg=cfg,
                logger=logger,
                rule_stock=stock,
                use_score_logic=uses_score_logic_feature(cfg, stock, side),
            )
            artifact["augmentation_state"] = augmentation_state
            artifact["source_train_file"] = cfg["files"]["train_by_stock"][stock]
            save_pickle(stock_model_path(stock, side), artifact)

            trades_df, signals_df = apply_lightgbm_artifact(
                full_panel=second_panel,
                raw_feat_df=augmented,
                artifact=artifact,
                stock=stock,
                side=side,
                minutes=all_minutes,
                dataset_split="train",
            )
            trades_all.append(trades_df)
            signals_all.append(signals_df)

            importance_df = artifact["feature_importance"].copy()
            importance_df.insert(0, "side", side)
            importance_df.insert(0, "stock", stock)
            feature_importances_all.append(importance_df)

            validation_rows = pd.concat([validation_grid, _artifact_model_row(artifact)], ignore_index=True, sort=False)
            validation_rows_all.append(validation_rows)
            manifest_rows.append(
                {
                    "stock": stock,
                    "side": side,
                    "model_path": str(stock_model_path(stock, side)),
                    "param_set": artifact["param_set"],
                    "threshold_quantile": float(artifact["threshold_quantile"]),
                    "threshold": float(artifact["threshold"]),
                    "best_iteration": int(artifact["best_iteration"]),
                    "rule_template_stock": artifact["rule_stock"],
                    "use_score_logic": bool(artifact["use_score_logic"]),
                    "features": artifact["model_features"],
                }
            )

    trades_df = pd.concat(trades_all, ignore_index=True)
    signals_df = pd.concat(signals_all, ignore_index=True)
    feature_importance_df = pd.concat(feature_importances_all, ignore_index=True)
    validation_df = pd.concat(validation_rows_all, ignore_index=True)
    manifest_df = pd.DataFrame(manifest_rows)
    _write_bundle(
        output_dir=output_dir,
        trades_df=trades_df,
        signals_df=signals_df,
        feature_importance_df=feature_importance_df,
        validation_df=validation_df,
        extras={"model_manifest.csv": manifest_df},
    )
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    return output_dir


def _load_existing_artifacts() -> dict[tuple[str, str], dict]:
    artifacts: dict[tuple[str, str], dict] = {}
    for stock in KNOWN_STOCKS:
        for side in ("buy", "sell"):
            path = stock_model_path(stock, side)
            if path.exists():
                artifacts[(stock, side)] = load_pickle(path)
    return artifacts


def _evaluate_known_stock_split(cfg: dict, split_label: str, file_map: dict[str, Path], artifacts: dict[tuple[str, str], dict], logger) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    frames = build_feature_frames_from_paths(cfg, file_map)
    trades_all: list[pd.DataFrame] = []
    signals_all: list[pd.DataFrame] = []
    total = len(file_map) * 2
    step = 0
    available_features = _available_features(cfg)
    for stock in file_map:
        base_df = frames[stock].copy()
        for side in ("buy", "sell"):
            artifact = artifacts.get((stock, side))
            if artifact is None:
                continue
            step += 1
            print(f"{progress_bar(step, total)} EXECUTION_ALGO {split_label} {stock} {side}", flush=True)
            selected_features = resolve_feature_set(cfg, available_features, stock, side)
            augmented = augment_feature_frame(base_df, artifact["augmentation_state"])
            second_panel = build_second_panel(
                augmented,
                selected_features=selected_features,
                top_decile=float(cfg["lightgbm"]["top_decile"]),
            )
            second_panel["target_return_bps"] = ridge_minute_return_series(second_panel, side)
            minutes = sorted(pd.to_datetime(base_df["minute"]).drop_duplicates().tolist())
            trades_df, signals_df = apply_lightgbm_artifact(
                full_panel=second_panel,
                raw_feat_df=augmented,
                artifact=artifact,
                stock=stock,
                side=side,
                minutes=minutes,
                dataset_split=split_label,
            )
            trades_all.append(trades_df)
            signals_all.append(signals_df)
    return trades_all, signals_all


def _fit_aapl_autonomous(cfg: dict, logger) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], dict[str, Any] | None]:
    train_name = cfg["files"].get("aapl_train_file")
    test_name = cfg["files"].get("aapl_test_file")
    if not train_name or not test_name:
        return [], [], [], [], None

    aapl_train_path = INPUT_DIR / str(train_name)
    aapl_test_path = INPUT_DIR / str(test_name)
    if not aapl_train_path.exists() or not aapl_test_path.exists():
        raise FileNotFoundError("AAPL train/test file configured but not found in execution_algorithm/input.")

    available_features = _available_features(cfg)
    train_frames = build_feature_frames_from_paths(cfg, {"AAPL": aapl_train_path})
    test_frames = build_feature_frames_from_paths(cfg, {"AAPL": aapl_test_path})
    base_train_df = train_frames["AAPL"].copy()
    base_test_df = test_frames["AAPL"].copy()
    inner_train_ratio = float(cfg["split"]["train_frac"]) / max(
        float(cfg["split"]["train_frac"]) + float(cfg["split"]["val_frac"]),
        1e-12,
    )
    train_df, val_df = split_train_val_only(base_train_df, train_frac_within_history=inner_train_ratio)
    train_minutes = sorted(pd.to_datetime(train_df["minute"]).drop_duplicates().tolist())
    val_minutes = sorted(pd.to_datetime(val_df["minute"]).drop_duplicates().tolist())
    all_train_minutes = sorted(pd.to_datetime(base_train_df["minute"]).drop_duplicates().tolist())
    test_minutes = sorted(pd.to_datetime(base_test_df["minute"]).drop_duplicates().tolist())

    trades_all: list[pd.DataFrame] = []
    signals_all: list[pd.DataFrame] = []
    feature_importances: list[pd.DataFrame] = []
    validations: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_artifacts: dict[str, dict] = {}

    for side in ("buy", "sell"):
        best_candidate: dict[str, Any] | None = None
        for source_stock in cfg["aapl"]["candidate_source_stocks"]:
            selected_features = resolve_feature_set(cfg, available_features, source_stock, side)
            use_score_logic = uses_score_logic_feature(cfg, source_stock, side)
            selection_aug_state = fit_augmentation_state(train_df)
            selection_augmented = augment_feature_frame(base_train_df, selection_aug_state)
            selection_panel = build_second_panel(
                selection_augmented,
                selected_features=selected_features,
                top_decile=float(cfg["lightgbm"]["top_decile"]),
            )
            selection_panel["target_return_bps"] = ridge_minute_return_series(selection_panel, side)
            artifact, validation_grid = fit_lightgbm_artifact(
                full_panel=selection_panel,
                raw_feat_df=selection_augmented,
                stock="AAPL",
                side=side,
                selected_features=selected_features,
                train_minutes=train_minutes,
                val_minutes=val_minutes,
                cfg=cfg,
                logger=logger,
                rule_stock=source_stock,
                use_score_logic=use_score_logic,
            )
            candidate_rows.append(
                {
                    "stock": "AAPL",
                    "side": side,
                    "source_stock": source_stock,
                    "objective": float(artifact["objective"]),
                    "param_set": artifact["param_set"],
                    "threshold_quantile": float(artifact["threshold_quantile"]),
                    "best_iteration": int(artifact["best_iteration"]),
                    "features": ",".join(artifact["model_features"]),
                    "use_score_logic": bool(artifact["use_score_logic"]),
                }
            )
            if best_candidate is None or float(artifact["objective"]) > float(best_candidate["objective"]):
                best_candidate = {
                    "source_stock": source_stock,
                    "selected_features": list(selected_features),
                    "use_score_logic": bool(use_score_logic),
                    "objective": float(artifact["objective"]),
                }

        assert best_candidate is not None
        final_aug_state = fit_augmentation_state(base_train_df)
        final_augmented = augment_feature_frame(base_train_df, final_aug_state)
        final_panel = build_second_panel(
            final_augmented,
            selected_features=best_candidate["selected_features"],
            top_decile=float(cfg["lightgbm"]["top_decile"]),
        )
        final_panel["target_return_bps"] = ridge_minute_return_series(final_panel, side)
        best_artifact, best_validation_grid = fit_lightgbm_artifact(
            full_panel=final_panel,
            raw_feat_df=final_augmented,
            stock="AAPL",
            side=side,
            selected_features=best_candidate["selected_features"],
            train_minutes=train_minutes,
            val_minutes=val_minutes,
            cfg=cfg,
            logger=logger,
            rule_stock=str(best_candidate["source_stock"]),
            use_score_logic=bool(best_candidate["use_score_logic"]),
        )
        best_artifact["augmentation_state"] = final_aug_state
        selected_artifacts[side] = best_artifact
        save_pickle(stock_model_path("AAPL", side), best_artifact)
        validations.append(pd.concat([best_validation_grid, _artifact_model_row(best_artifact)], ignore_index=True, sort=False))

        train_augmented = augment_feature_frame(base_train_df, best_artifact["augmentation_state"])
        train_panel = build_second_panel(
            train_augmented,
            selected_features=best_artifact["selected_features"],
            top_decile=float(cfg["lightgbm"]["top_decile"]),
        )
        train_panel["target_return_bps"] = ridge_minute_return_series(train_panel, side)
        train_trades_df, train_signals_df = apply_lightgbm_artifact(
            full_panel=train_panel,
            raw_feat_df=train_augmented,
            artifact=best_artifact,
            stock="AAPL",
            side=side,
            minutes=all_train_minutes,
            dataset_split="train",
        )
        trades_all.append(train_trades_df)
        signals_all.append(train_signals_df)

        test_augmented = augment_feature_frame(base_test_df, best_artifact["augmentation_state"])
        test_panel = build_second_panel(
            test_augmented,
            selected_features=best_artifact["selected_features"],
            top_decile=float(cfg["lightgbm"]["top_decile"]),
        )
        test_panel["target_return_bps"] = ridge_minute_return_series(test_panel, side)
        test_trades_df, test_signals_df = apply_lightgbm_artifact(
            full_panel=test_panel,
            raw_feat_df=test_augmented,
            artifact=best_artifact,
            stock="AAPL",
            side=side,
            minutes=test_minutes,
            dataset_split="test",
        )
        trades_all.append(test_trades_df)
        signals_all.append(test_signals_df)

        importance_df = best_artifact["feature_importance"].copy()
        importance_df.insert(0, "side", side)
        importance_df.insert(0, "stock", "AAPL")
        feature_importances.append(importance_df)

    return trades_all, signals_all, feature_importances, validations, {"candidates": pd.DataFrame(candidate_rows)}


def run_test_mode(cfg: dict | None = None) -> Path:
    """Replay trained 4-stock models on new tests and run the autonomous AAPL path if present."""
    cfg = cfg or execution_config()
    ensure_dirs()
    set_seed(int(cfg["seed"]))
    output_dir = OUTPUT_DIR / "test"
    logger = make_logger("execution_algorithm_test", output_dir / "run.log")
    artifacts = _load_existing_artifacts()
    if len(artifacts) < len(KNOWN_STOCKS) * 2:
        raise FileNotFoundError("Missing one or more trained 4-stock model artifacts in execution_algorithm/model.")

    trades_all: list[pd.DataFrame] = []
    signals_all: list[pd.DataFrame] = []
    feature_importances_all: list[pd.DataFrame] = []
    validations_all: list[pd.DataFrame] = []
    extras: dict[str, pd.DataFrame] = {}

    train_file_map = {
        stock: INPUT_DIR / filename
        for stock, filename in cfg["files"]["train_by_stock"].items()
        if filename is not None and (INPUT_DIR / filename).exists()
    }
    train_trades, train_signals = _evaluate_known_stock_split(cfg, "train", train_file_map, artifacts, logger)
    trades_all.extend(train_trades)
    signals_all.extend(train_signals)

    for (stock, side), artifact in artifacts.items():
        importance_df = artifact["feature_importance"].copy()
        importance_df.insert(0, "side", side)
        importance_df.insert(0, "stock", stock)
        feature_importances_all.append(importance_df)
        validations_all.append(pd.concat([artifact["validation_grid"], _artifact_model_row(artifact)], ignore_index=True, sort=False))

    test_file_map = {
        stock: INPUT_DIR / filename
        for stock, filename in cfg["files"]["test_by_stock"].items()
        if filename is not None and (INPUT_DIR / filename).exists()
    }
    if test_file_map:
        test_trades, test_signals = _evaluate_known_stock_split(cfg, "test", test_file_map, artifacts, logger)
        trades_all.extend(test_trades)
        signals_all.extend(test_signals)

    aapl_trades, aapl_signals, aapl_importances, aapl_validations, aapl_extras = _fit_aapl_autonomous(cfg, logger)
    trades_all.extend(aapl_trades)
    signals_all.extend(aapl_signals)
    feature_importances_all.extend(aapl_importances)
    validations_all.extend(aapl_validations)
    if aapl_extras:
        extras["aapl_template_selection.csv"] = aapl_extras["candidates"]

    trades_df = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    signals_df = pd.concat(signals_all, ignore_index=True) if signals_all else pd.DataFrame()
    feature_importance_df = pd.concat(feature_importances_all, ignore_index=True) if feature_importances_all else pd.DataFrame()
    validation_df = pd.concat(validations_all, ignore_index=True) if validations_all else pd.DataFrame()
    _write_bundle(
        output_dir=output_dir,
        trades_df=trades_df,
        signals_df=signals_df,
        feature_importance_df=feature_importance_df,
        validation_df=validation_df,
        extras=extras,
    )
    return output_dir

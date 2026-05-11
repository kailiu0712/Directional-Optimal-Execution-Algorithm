# Execution Algorithm Package

This directory is a self-contained, deployable copy of the current best LightGBM execution system. The only non-stdlib dependencies are the same external packages already used by the baseline workflow: `numpy`, `pandas`, and `lightgbm`.

## Goal

The package supports two operational modes:

- `train`: train the four known-stock execution algorithms on the available training data inside `input/`, save the fitted models to `model/`, and report full in-sample results.
- `test`: load the saved four-stock models from `model/`, run them on configured test files, and also run the autonomous AAPL pipeline if `AAPL` train and test files are provided.

The package is designed so that:

- running `train` on the copied four-stock training files reproduces the full in-sample results;
- running `test` uses only pre-trained four-stock models on their test files;
- running `test` for `AAPL` performs automatic template selection and threshold tuning from AAPL training data only, then evaluates on AAPL test data.

## Directory Layout

```text
execution_algorithm/
  input/
    AMZN_5levels_train.csv
    GOOG_5levels_train.csv
    INTC_5levels_train.csv
    MSFT_5levels_train.csv
    ... add future test files here ...
  model/
    AMZN_buy.pkl
    AMZN_sell.pkl
    ...
  output/
    train/
    test/
  common.py
  config.py
  data.py
  data_utils.py
  features.py
  panel.py
  score_logic.py
  lightgbm_engine.py
  pipeline.py
  run.py
```

## How To Run

From the repo root:

```powershell
python execution_algorithm\run.py
```

The package reads `mode` from [config.py](./config.py).

### Train Mode

Set:

```python
"mode": "train"
```

This will:

1. load the four stock train files from `input/`;
2. rebuild the feature panel;
3. train one buy model and one sell model for each stock;
4. save the eight fitted artifacts to `model/`;
5. write in-sample outputs to `output/train/`.

Main outputs:

- `trades.csv`
- `summary.csv`
- `cost_improvement_metrics.csv`
- `signals.csv`
- `feature_importance.csv`
- `validation_selection.csv`
- `model_manifest.csv`

### Test Mode

Set:

```python
"mode": "test"
```

Then edit the file mappings in [config.py](./config.py):

- `files.test_by_stock`
- `files.aapl_train_file`
- `files.aapl_test_file`

This will:

1. load the saved four-stock artifacts from `model/`;
2. evaluate those artifacts on the extant four-stock train files and any configured four-stock test files;
3. if AAPL train and test files are both supplied, run the autonomous AAPL algorithm;
4. write outputs to `output/test/`.

## Metrics Reported

### Trade Summary

`summary.csv` reports, by `stock` and `side`:

- `avg_improvement`
- `median_improvement`
- `std_improvement`
- `theoretical_best_avg_improvement`
- `avg_exec_sec_into_minute`
- `win_rate`
- `model_pct_of_theoretical_best`

`model_pct_of_theoretical_best` is:

```text
avg_improvement / theoretical_best_avg_improvement
```

### Cost Improvement Vs TWAP

`cost_improvement_metrics.csv` reports the requested metric:

```text
100 - 100 * (TOTAL_YOURALGO_BUY - TOTAL_YOURALGO_SELL)
          / (TOTAL_TWAP_BUY - TOTAL_TWAP_SELL)
```

Interpretation:

- `0%` means the algorithm matches TWAP.
- `70%` means the algorithm reduces execution cost to `30%` of TWAP cost.

The file is written by dataset split:

- `train`
- `test`

and also includes an `ALL` stock aggregate for each split.

## Pipeline Details

### 1. Feature Engineering

`data_utils.py` builds the raw 100ms feature panel, including:

- spread and microprice features
- book imbalance and depth features
- rolling momentum, z-score, and range features
- event counts and “seconds since event” features
- cross-stock features for the original four stocks

No future information is used in feature construction.

### 2. Residual / Interaction Features

`features.py` adds the current best additional feature augmentations:

- `pressure_term_structure`
- `depth_slope_bid_resid`
- `microprice_edge_resid`
- `time_book_pressure3_interact`
- `time_ask_top_share1_interact`

These residualized features are calibrated from training data only, then applied unchanged at inference time.

### 3. Second-Level Panel

`panel.py` aggregates 100ms rows into per-second rows inside each minute and builds the within-minute target structure used by the LightGBM model.

### 4. Score-Logic Overlay

`score_logic.py` contains a rule-based baseline signal.

Important:

- score-logic thresholds are fitted on training data only;
- test data never re-fits score-logic thresholds;
- models with disabled score logic keep it disabled in deployment as well.

### 5. LightGBM Selection

`lightgbm_engine.py` implements:

- model fitting
- threshold-quantile selection
- score-logic application
- gate-rule application
- execution-row selection

The current logic is:

1. fit candidate LightGBM parameter sets on training data;
2. evaluate them on validation data using the current execution objective;
3. choose the best parameter set and threshold quantile;
4. refit on train plus validation if configured;
5. save the fitted artifact for deployment.

### 6. Execution Rule

Within each minute, execution chooses the **first** second where:

- `gate_pass >= 1`
- `signal_score >= threshold`

If no such second exists, it falls back to the last second of the minute.

### 7. Leakage Controls

This package emphasize strict leakage controls:

- feature construction uses only contemporaneous or past observations;
- residual-feature calibration uses training data only;
- score-logic threshold fitting uses training data only;
- execution gate thresholds use training data only;
- threshold quantile selection uses validation only;
- inference on test files uses saved model artifacts without any re-fitting on test data.

## Four-Stock Models

The four-stock deployable system includes eight models:

- `AMZN_buy`
- `AMZN_sell`
- `GOOG_buy`
- `GOOG_sell`
- `INTC_buy`
- `INTC_sell`
- `MSFT_buy`
- `MSFT_sell`

Their feature sets and score-logic usage are frozen to the current best baseline in [config.py](./config.py).

## AAPL Autonomous Logic

When `AAPL` train and test files are present in `test` mode:

1. the package builds AAPL features from AAPL training data only;
2. for each side, it tries the existing buy or sell templates from:
   - `AMZN`
   - `GOOG`
   - `INTC`
   - `MSFT`
3. it selects the best template by AAPL validation objective;
4. it refits the chosen template on all AAPL training data;
5. it evaluates on AAPL test data.

There is no hand-tuning specific to AAPL.

The AAPL selection table is written to:

- `output/test/aapl_template_selection.csv`

## Key Config Parameters

In [config.py](./config.py):

- `mode`: `train` or `test`
- `files.train_by_stock`: training filenames for the original four stocks
- `files.test_by_stock`: future test filenames for the original four stocks
- `files.aapl_train_file`: AAPL training filename
- `files.aapl_test_file`: AAPL test filename
- `lightgbm.threshold_quantile_grid`: threshold candidates
- `lightgbm.candidate_param_sets`: LightGBM parameter candidates
- `lightgbm.feature_overrides_by_model`: frozen best per-model feature sets
- `lightgbm.disable_score_logic_by_model`: models that should not use score logic

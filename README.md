# Smart Execution Algorithm — LightGBM Within-Minute Timing

Eight independent LightGBM models (one per stock × side) time a single execution
per minute to beat the TWAP benchmark on 5-level limit-order-book data.

---

## Achievements

### Out-of-Sample Performance (120 test minutes per stock)

| Stock | Buy \$/trade | Sell \$/trade | Win % | vs TWAP |
|-------|-------------|--------------|-------|---------|
| AMZN  | +0.018      | +0.018       | 51 %  | **31.5 %** |
| GOOG  | +0.037      | +0.030       | 45 %  | **30.5 %** |
| INTC  | +0.001      | +0.000       | 23 %  | **11.0 %** |
| MSFT  | +0.003      | +0.002       | 41 %  | **50.4 %** |
| AAPL  | +0.019      | +0.011       | 49 %  | **24.7 %** |
| **ALL** | **+0.015** | **+0.012** | **43 %** | **29.3 %** |

**Round-trip cost reduction vs TWAP: 29.3 % out-of-sample across all 5 stocks.**
The algorithm saves **\$16.64** of the \$56.74 round-trip execution cost that TWAP
incurs over 120 test minutes — without any look-ahead or post-hoc tuning.

### In-Sample Performance (270 training minutes, 4 stocks)

| Stock | Algo \$ | TWAP \$ | **Saving** |
|-------|---------|---------|-----------|
| AMZN  | 19.08   | 37.86   | **49.6 %** |
| GOOG  | 46.67   | 78.93   | **40.9 %** |
| INTC  |  0.85   |  2.75   | **69.0 %** |
| MSFT  |  0.24   |  2.75   | **91.4 %** |
| **ALL** | **66.84** | **122.29** | **45.3 %** |

The system captures **28 %** of the theoretical best improvement (best hindsight
execution), averaged across all 5 stocks and both sides.

---

## Overview

The package supports two operational modes:

- **`train`** — train the four known-stock models on `input/` data, save fitted
  artifacts to `model/`, and report full in-sample results.
- **`test`** — load saved models, evaluate on test files, and autonomously handle
  any new stock (AAPL) using template selection from training data only.

---

## Directory Layout

```text
execution_algorithm/
  input/
    AMZN_5levels_train.csv
    GOOG_5levels_train.csv
    INTC_5levels_train.csv
    MSFT_5levels_train.csv
    AAPL_5levels_train.csv   ← added at test time
    AAPL_5levels_test.csv    ← added at test time
    ... other test files ...
  model/
    AMZN_buy.pkl  AMZN_sell.pkl
    GOOG_buy.pkl  GOOG_sell.pkl
    INTC_buy.pkl  INTC_sell.pkl
    MSFT_buy.pkl  MSFT_sell.pkl
  output/
    train/
    test/
      plots/
  common.py       config.py      data.py
  data_utils.py   features.py    panel.py
  score_logic.py  lightgbm_engine.py
  pipeline.py     run.py
```

---

## How To Run

From the **project root** (not inside `execution_algorithm/`):

```powershell
python -m execution_algorithm.run
```

The active mode is set in [`config.py`](./config.py).

### Train Mode

```python
"mode": "train"
```

1. Loads the four stock train files from `input/`.
2. Builds the 100+ feature panel and second-level label.
3. Trains one buy model and one sell model per stock (8 total).
4. Calibrates threshold quantile on all training minutes for each model.
5. Saves the eight fitted artifacts to `model/`.
6. Writes in-sample outputs to `output/train/`.

**Outputs:** `trades.csv`, `summary.csv`, `cost_improvement_metrics.csv`,
`signals.csv`, `feature_importance.csv`, `validation_selection.csv`,
`model_manifest.csv` / `model_manifest.json`.

### Test Mode

```python
"mode": "test"
```

Edit file mappings in [`config.py`](./config.py):

```python
"test_by_stock":   {"AMZN": "...", "GOOG": "...", ...}
"aapl_train_file": "AAPL_5levels_train.csv"
"aapl_test_file":  "AAPL_5levels_test.csv"
```

1. Loads saved four-stock artifacts from `model/`.
2. Evaluates those artifacts on training files (in-sample replay) and on any
   configured test files.
3. If both AAPL files are present, runs the autonomous AAPL pipeline.
4. Writes all outputs to `output/test/`, including a `plots/` subdirectory.

---

## Algorithm Design

### Training Label — Intra-Minute Regret

For each second `t` within a minute the label measures how much better the
current price is than the best price remaining later in that same minute:

```
y_buy(t)  = 1e4 * (min_{s>t} ask_s  - ask_t)  / ask_t
y_sell(t) = 1e4 * (bid_t   - max_{s>t} bid_s) / bid_t
```

Positive = current price is the best remaining; execute now.
No future information crosses the minute boundary.

### Feature Engineering (100+ features, 8–9 selected per model)

| Group | Count | Examples |
|-------|-------|---------|
| Spread & liquidity | 4 | `spread_bps`, `spread_mean_roll_1s` |
| Book pressure/imbalance | 6 | `book_pressure_1`, `book_pressure_5` |
| Depth structure | 21 | `depth_ratio_5`, `depth_sum_ask` |
| Price momentum & z-score | 21 | `momentum_10s_bps`, `mid_zscore_10s` |
| Microprice & range | 6 | `microprice_edge_bps`, `mid_pos_in_range_10s` |
| Trade/cancel pressure | 12 | `bid_down_count_10s` |
| Quote event counts | 24 | `ask_up_count_10s`, `ask_up_count_3s` |
| Time-since-event | 5 | `secs_since_ask_down` |
| Cross-stock OFI | 5 | `cross_buy_pressure_mean` |

**Augmented features** (OLS-residualised to isolate independent alpha):

- `depth_slope_bid_resid` — bid-side book shape net of size level
- `microprice_edge_resid` — tick position net of L1 imbalance
- `pressure_term_structure` — L1 minus L3 imbalance gradient
- `time_book_pressure3_interact`, `time_ask_top_share1_interact` — MSFT only

Residual calibration uses training data only and is applied unchanged at
inference time.

### Score Logic Signal

A hand-crafted rule score (weighted sum of quantile-threshold indicators on
spread, depth, and momentum) is computed per second, EWM-smoothed (span = 3),
and fed as `score_logic_signal` **into LightGBM as a feature** — not as a hard
gate. This lets the model learn when to trust and when to override the rule.

A hard spread gate additionally blocks wide-spread seconds for AMZN and GOOG.
Score logic is disabled entirely for INTC sell and MSFT buy (depth-driven models
where the spread-based rule adds noise).

### LightGBM Model & Threshold Calibration

All eight models use `depth4_balanced` hyperparameters:
`max_depth=4`, `num_leaves=15`, `lr=0.035`, `n_estimators=450`,
`reg_lambda=5`, `reg_alpha=0.75`.

Threshold calibration:

1. Grid-search `q* ∈ {0.08, 0.12, 0.16, 0.20, 0.24, 0.30}`.
2. Score each candidate with `L = mean_improvement - 0.01*forced_last_rate - 0.05*std_improvement`.
3. **Calibrate on all training minutes** (train = val), consistently for all 5
   stocks including AAPL.

### Execution Rule

Within each minute, execute at the **first second** where:
- gate check passes (if applicable), **and**
- `signal_score >= threshold`

**Fallback:** if no second qualifies, execute at the last second of the minute
(forced-last).

### Leakage Controls

- Feature construction uses only contemporaneous or past observations.
- All residual-feature and score-logic calibrations use training data only.
- Threshold selection uses training data only (no test data ever seen before inference).
- Test inference uses saved artifacts without any re-fitting.

---

## Four-Stock Models

| Model | Top features | Gate | Score logic |
|-------|-------------|------|-------------|
| AMZN buy  | `time_in_minute`, `spread_bps`, `spread_mean_roll_1s` | spread > 30th pct | ✓ |
| AMZN sell | `spread_bps`, `spread_mean_roll_1s`, `score_logic_signal` | spread > 35th pct | ✓ |
| GOOG buy  | `spread_bps`, `depth_slope_ask`, `ask_top_share_3_of_5` | spread > 25th pct | ✓ |
| GOOG sell | `spread_bps`, `time_in_minute`, `momentum_10s_bps` | spread > 25th pct | ✓ |
| INTC buy  | `depth_slope_bid_resid`, `BidSizeLog_1`, `depth_ratio_5` | none | ✓ |
| INTC sell | `depth_sum_ask`, `AskSizeLog_3`, `depth_conc_diff_1_5` | none | ✗ |
| MSFT buy  | `BidSizeLog_5`, `time_in_minute`, `ask_top_share_1_of_5` | none | ✗ |
| MSFT sell | `BidSizeLog_3`, `book_pressure_3`, `slope_diff_5` | none | ✓ |

AMZN/GOOG are **spread-driven** (execute when spread compresses).
INTC/MSFT are **depth-driven** (execute when own-side queue is large).

---

## AAPL Autonomous Pipeline

AAPL is not in the original training set. When both AAPL train and test files
are present in test mode, the pipeline:

1. Builds AAPL features from AAPL training data only.
2. For each side, trains a candidate model under each of the four source-stock
   feature templates (AMZN, GOOG, INTC, MSFT), using **all AAPL training
   minutes** for both fitting and threshold calibration.
3. Selects the template with the highest validation objective.
4. Refits the chosen template on all AAPL training data.
5. Evaluates on AAPL test data with no further tuning.

**Selected templates (current run):**

| Side | Template | Val obj | Runner-up |
|------|---------|---------|-----------|
| Buy  | INTC    | 0.068   | GOOG (0.041) |
| Sell | MSFT    | 0.055   | INTC (0.055) |

AAPL OOS result: **+24.7 % vs TWAP** (buy +\$0.019/trade, sell +\$0.011/trade).
Template selection log: `output/test/aapl_template_selection.csv`.

---

## Output Files Reference

| File | Contents |
|------|---------|
| `trades.csv` | One row per executed trade (price, improvement, timing, signal) |
| `summary.csv` | Per-stock/side averages: improvement, win rate, % of theoretical best |
| `cost_improvement_metrics.csv` | Round-trip cost reduction vs TWAP by split and stock |
| `signals.csv` | Full second-level signal panel for all minutes |
| `feature_importance.csv` | LightGBM gain-based importance per model |
| `validation_selection.csv` | Full threshold grid results for all models |
| `model_manifest.csv/json` | Frozen artifact metadata for each of the 8 models |
| `aapl_template_selection.csv` | Per-side candidate template scores for AAPL |
| `output/test/plots/` | Four diagnostic plots (cumulative improvement, scatter, sampled minutes) |

---

## Key Config Parameters (`config.py`)

| Parameter | Description |
|-----------|-------------|
| `mode` | `"train"` or `"test"` |
| `files.train_by_stock` | Training CSV filenames for AMZN/GOOG/INTC/MSFT |
| `files.test_by_stock` | Test CSV filenames for AMZN/GOOG/INTC/MSFT |
| `files.aapl_train_file` | AAPL training CSV filename |
| `files.aapl_test_file` | AAPL test CSV filename |
| `lightgbm.threshold_quantile_grid` | Threshold candidates for calibration |
| `lightgbm.candidate_param_sets` | LightGBM hyperparameter sets to try |
| `lightgbm.feature_overrides_by_model` | Frozen best feature sets per model |
| `lightgbm.disable_score_logic_by_model` | Models that do not use score logic |
| `aapl.candidate_source_stocks` | Templates evaluated for AAPL selection |

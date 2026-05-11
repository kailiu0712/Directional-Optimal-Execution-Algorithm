from __future__ import annotations

import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PACKAGE_DIR = Path(__file__).resolve().parent
INPUT_DIR = PACKAGE_DIR / "input"
MODEL_DIR = PACKAGE_DIR / "model"
OUTPUT_DIR = PACKAGE_DIR / "output"

KNOWN_STOCKS = ("AMZN", "GOOG", "INTC", "MSFT")
SUPPORTED_STOCKS = KNOWN_STOCKS + ("AAPL",)
GROUPS = {
    "AMZN_GOOG": ["AMZN", "GOOG"],
    "INTC_MSFT": ["INTC", "MSFT"],
}
STOCK_TO_GROUP = {stock: group for group, stocks in GROUPS.items() for stock in stocks}


def ensure_dirs() -> None:
    for path in (INPUT_DIR, MODEL_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def progress_bar(current: int, total: int, width: int = 24) -> str:
    filled = int(width * current / max(total, 1))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def make_logger(name: str, path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def stock_model_path(stock: str, side: str) -> Path:
    return MODEL_DIR / f"{stock}_{side}.pkl"


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)

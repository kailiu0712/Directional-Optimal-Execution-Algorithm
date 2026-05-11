from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from execution_algorithm.config import execution_config
    from execution_algorithm.pipeline import run_test_mode, train_known_stock_models
else:
    from .config import execution_config
    from .pipeline import run_test_mode, train_known_stock_models


def main() -> None:
    cfg = execution_config()
    mode = str(cfg["mode"]).strip().lower()
    if mode == "train":
        output_dir = train_known_stock_models(cfg)
    elif mode == "test":
        output_dir = run_test_mode(cfg)
    else:
        raise ValueError("config['mode'] must be either 'train' or 'test'.")
    print(f"Execution algorithm outputs saved under {output_dir}", flush=True)


if __name__ == "__main__":
    main()

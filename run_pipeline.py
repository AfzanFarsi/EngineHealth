"""One command that does everything: train -> explain -> time-series -> figures.

    python run_pipeline.py --subset FD001        # just the baseline (about 7 minutes)
    python run_pipeline.py --all                 # all four subsets (about 45 minutes)
"""
import argparse
import time

import config as cfg
from src import explainability, timeseries, train, visualization


def main():
    ap = argparse.ArgumentParser(description="EngineHealth full pipeline")
    ap.add_argument("--subset", choices=list(cfg.SUBSETS), default=cfg.DEFAULT_SUBSET)
    ap.add_argument("--all", action="store_true", help="run FD001, FD002, FD003 and FD004")
    a = ap.parse_args()
    t0 = time.time()
    for s in (list(cfg.SUBSETS) if a.all else [a.subset]):
        train.run_subset(s)
        explainability.run(s)
        timeseries.run(s)
        visualization.save_subset_figures(s)
        print(f"[{s}] figures saved to {cfg.FIGURES_DIR / s}", flush=True)
    visualization.save_cross_subset_figures()
    print(f"Pipeline finished in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()

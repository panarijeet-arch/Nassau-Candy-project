"""
run_pipeline.py
================
One-time orchestration script: loads raw data, cleans it, engineers
features, trains & compares models, runs clustering, and caches the
results to disk (models/ and data/processed_features.parquet) so the
Streamlit app can load instantly instead of retraining on every launch.

Run this once after cloning the repo (also run automatically by app.py
on first launch if the cache is missing):

    python src/run_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import load_raw, run_data_quality_report, clean_data
from feature_engineering import build_feature_table
from model_training import train_and_compare, select_best_model
from clustering import cluster_routes
from utils import save_pipeline, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
RAW_CSV = DATA_DIR / "Nassau_Candy_Distributor.csv"


def main() -> None:
    logger.info("=== Nassau Candy AI Pipeline: starting ===")

    df_raw = load_raw(RAW_CSV)
    dq_report = run_data_quality_report(df_raw)
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "data_quality_report.json", "w") as f:
        json.dump(dq_report, f, indent=2, default=str)
    logger.info("Data quality report saved to reports/data_quality_report.json")

    df_clean = clean_data(df_raw)
    features = build_feature_table(df_clean)

    clustered, cluster_meta = cluster_routes(features)
    with open(REPORTS_DIR / "cluster_meta.json", "w") as f:
        json.dump(cluster_meta, f, indent=2, default=str)

    results = train_and_compare(clustered)
    best_name, best, explanation = select_best_model(results)
    logger.info("Best model: %s", best_name)
    logger.info(explanation)

    model_comparison = {
        name: {"rmse": r.rmse, "mae": r.mae, "r2": r.r2,
               "cv_r2_mean": r.cv_r2_mean, "cv_r2_std": r.cv_r2_std}
        for name, r in results.items()
    }
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump({"models": model_comparison, "best_model": best_name,
                   "explanation": explanation}, f, indent=2)

    save_pipeline(best.pipeline, name="best_model")
    with open(MODELS_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)

    clustered.to_csv(DATA_DIR / "processed_features.csv", index=False)
    logger.info("Processed feature table saved to data/processed_features.csv")
    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()

"""
train_pipeline.py
-----------------
Entry point for the full training workflow.

Usage:
    python -m src.pipeline.train_pipeline --data path/to/data.csv
"""

import argparse
import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logger

DEFAULT_DATA_PATH = os.path.join(
    "data", "fmcg_sales_marketing_profitability_2023_2025.csv"
)


def run_training(data_path: str = DEFAULT_DATA_PATH) -> None:
    logger.info("=== Training pipeline started ===")
    try:
        # 1. Ingest
        ingestion = DataIngestion()
        train_path, test_path = ingestion.run(data_path)

        # 2. Transform
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test = transformation.run(train_path, test_path)

        # 3. Train
        trainer = ModelTrainer()
        best_model, best_model_name, best_r2 = trainer.run(
            X_train, y_train, X_test, y_test
        )

        # 4. Evaluate
        # Derive numeric + categorical column names from the transformation config
        import pandas as pd
        from src.components.data_transformation import apply_feature_engineering, TARGET, LEAKAGE_COLS

        raw_df = pd.read_csv(train_path)
        fe_df  = apply_feature_engineering(raw_df)
        X_cols = fe_df.drop(columns=[TARGET]).columns.tolist()
        numeric_cols     = fe_df.drop(columns=[TARGET]).select_dtypes(include=["int64","float64"]).columns.tolist()
        categorical_cols = fe_df.drop(columns=[TARGET]).select_dtypes(include=["object"]).columns.tolist()
        feature_names    = numeric_cols + categorical_cols

        evaluator = ModelEvaluation()
        evaluator.run(
            preprocessor_path=os.path.join("artifacts", "preprocessor.pkl"),
            model_path=os.path.join("artifacts", "model.pkl"),
            test_path=test_path,
            feature_names=feature_names,
        )

        logger.info("=== Training pipeline complete ===")
        print(f"\nTraining complete. Best model: {best_model_name}  R²={best_r2}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FMCG Profit — Training Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the raw CSV dataset",
    )
    args = parser.parse_args()
    run_training(args.data)

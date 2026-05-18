import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import evaluate_models, save_object

warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge":             Ridge(alpha=1.0),
    "Lasso":             Lasso(alpha=0.1),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost":          AdaBoostRegressor(n_estimators=100, random_state=42),
    "SVR":               SVR(kernel="rbf"),
    "KNN":               KNeighborsRegressor(n_neighbors=5),
}


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[object, str, float]:
        """
        Train all candidate models, compare by R², and save the best one.

        Returns
        -------
        (best_model, best_model_name, best_r2)
        """
        logger.info("Starting model training")
        try:
            results_df = evaluate_models(X_train, y_train, X_test, y_test, MODELS)

            print("\n=== Model Comparison (sorted by R²) ===")
            print(results_df.to_string(index=False))

            best_model_name = results_df.iloc[0]["Model"]
            best_r2         = results_df.iloc[0]["R2"]
            best_model      = MODELS[best_model_name]

            # Re-fit the best model on the full training array
            # (evaluate_models already fitted it, but we make this explicit)
            best_model.fit(X_train, y_train)

            save_object(self.config.model_path, best_model)
            logger.info(
                f"Best model: {best_model_name} (R²={best_r2}). "
                f"Saved to {self.config.model_path}"
            )

            print(f"\nBest model: {best_model_name}  R² = {best_r2}")
            return best_model, best_model_name, best_r2

        except Exception as e:
            raise CustomException(e, sys)

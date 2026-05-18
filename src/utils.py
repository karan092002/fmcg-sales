import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

from src.exception import CustomException
from src.logger import logger


def save_object(filepath: str, obj) -> None:
    """Pickle an object to disk, creating parent directories as needed."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {filepath}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(filepath: str):
    """Load a pickled object from disk."""
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from {filepath}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Train each model and compute R², RMSE, MAE, CV Mean, and CV Std.

    Returns a DataFrame sorted by R² (descending).
    """
    try:
        results = []

        for name, model in models.items():
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2   = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae  = mean_absolute_error(y_test, preds)

            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1
            )

            results.append({
                "Model":   name,
                "R2":      round(r2, 4),
                "RMSE":    round(rmse, 2),
                "MAE":     round(mae, 2),
                "CV Mean": round(cv_scores.mean(), 4),
                "CV Std":  round(cv_scores.std(), 4),
            })
            logger.info(
                f"{name} — R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, "
                f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            )

        return (
            pd.DataFrame(results)
            .sort_values("R2", ascending=False)
            .reset_index(drop=True)
        )

    except Exception as e:
        raise CustomException(e, sys)

import os
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object

sns.set_theme()


@dataclass
class ModelEvaluationConfig:
    eval_dir:           str = os.path.join("artifacts", "evaluation")
    report_path:        str = field(init=False)
    actual_vs_pred_path: str = field(init=False)
    residuals_path:     str = field(init=False)
    feature_imp_path:   str = field(init=False)

    def __post_init__(self):
        self.report_path         = os.path.join(self.eval_dir, "report.txt")
        self.actual_vs_pred_path = os.path.join(self.eval_dir, "actual_vs_predicted.png")
        self.residuals_path      = os.path.join(self.eval_dir, "residuals.png")
        self.feature_imp_path    = os.path.join(self.eval_dir, "feature_importance.png")


class ModelEvaluation:
    def __init__(self):
        self.config = ModelEvaluationConfig()
        os.makedirs(self.config.eval_dir, exist_ok=True)

    def run(
        self,
        preprocessor_path: str,
        model_path: str,
        test_path: str,
        feature_names: list | None = None,
    ) -> None:
        """
        Load saved preprocessor and model, run evaluation on the test set,
        and write all plots and the text report to artifacts/evaluation/.
        """
        logger.info("Starting model evaluation")
        try:
            from src.components.data_transformation import apply_feature_engineering, TARGET

            preprocessor = load_object(preprocessor_path)
            model        = load_object(model_path)

            test_df  = pd.read_csv(test_path)
            test_df  = apply_feature_engineering(test_df)
            X_test   = test_df.drop(columns=[TARGET])
            y_test   = test_df[TARGET].values

            X_test_arr = preprocessor.transform(X_test)
            preds      = model.predict(X_test_arr)
            residuals  = y_test - preds

            r2   = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae  = mean_absolute_error(y_test, preds)

            # ── Text report ────────────────────────────────────────────────
            report_lines = [
                f"Model: {model.__class__.__name__}",
                f"R²:    {r2:.4f}",
                f"RMSE:  {rmse:.2f}",
                f"MAE:   {mae:.2f}",
                "",
                "Residual Summary:",
                str(pd.Series(residuals).describe().round(2)),
            ]
            report_text = "\n".join(report_lines)
            print(report_text)
            with open(self.config.report_path, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {self.config.report_path}")

            # ── Actual vs Predicted ────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test, preds, alpha=0.3, s=10)
            lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
            ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
            ax.set_xlabel("Actual Profit_USD")
            ax.set_ylabel("Predicted Profit_USD")
            ax.set_title(f"Actual vs Predicted — {model.__class__.__name__}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.config.actual_vs_pred_path, dpi=120)
            plt.close(fig)
            logger.info(f"Actual vs Predicted plot saved")

            # ── Residuals ──────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(preds, residuals, alpha=0.3, s=10)
            axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Residual")
            axes[0].set_title("Residuals vs Predicted")

            sns.histplot(residuals, bins=50, kde=True, ax=axes[1])
            axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
            axes[1].set_xlabel("Residual")
            axes[1].set_title("Residual Distribution")

            fig.tight_layout()
            fig.savefig(self.config.residuals_path, dpi=120)
            plt.close(fig)
            logger.info(f"Residual plots saved")

            # ── Feature Importance (if available) ──────────────────────────
            if hasattr(model, "feature_importances_") and feature_names:
                importances = pd.Series(
                    model.feature_importances_, index=feature_names
                ).sort_values(ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=importances.values, y=importances.index, ax=ax)
                ax.set_title(f"Feature Importances — {model.__class__.__name__}")
                ax.set_xlabel("Mean Decrease in Impurity")
                fig.tight_layout()
                fig.savefig(self.config.feature_imp_path, dpi=120)
                plt.close(fig)
                logger.info(f"Feature importance plot saved")

            logger.info("Model evaluation complete")

        except Exception as e:
            raise CustomException(e, sys)

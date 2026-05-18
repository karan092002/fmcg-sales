import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

TARGET = "Profit_USD"

# Columns that directly reveal or are derived from Profit_USD — must be dropped
# before any model sees the data.
LEAKAGE_COLS = [
    "Order_ID",
    "SKU",
    "Order_Date",
    "Gross_Sales_USD",
    "Net_Revenue_USD",
    "COGS_USD",
    "Logistics_Cost_USD",
    "Profit_Margin_Pct",
]


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features and drop leakage / identifier columns.

    This function is called identically in the transformation component and
    the predict pipeline, which guarantees the feature space is the same at
    inference time.
    """
    df = df.copy()

    # Effective price after discount
    df["Effective_Unit_Price"] = df["Unit_Price_USD"] * (1 - df["Discount_Pct"] / 100)

    # Estimated top-line revenue using only pre-sale decision variables
    df["Revenue_Estimate"] = df["Units_Sold"] * df["Effective_Unit_Price"]

    # Marketing spend per unit — a measure of marketing efficiency
    df["Marketing_Per_Unit"] = df["Marketing_Spend_USD"] / df["Units_Sold"]

    # Binary flag for any active promotion
    df["Is_Promoted"] = (df["Promotion_Type"] != "No Promo").astype(int)

    # Binary flag for the holiday quarter, which showed a profit lift in EDA
    df["Is_Q4"] = (df["Quarter"] == "Q4").astype(int)

    # Drop leakage and identifier columns (TARGET is not in LEAKAGE_COLS;
    # it is separated by the caller)
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def _build_preprocessor(self, numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
        """
        Construct a ColumnTransformer with:
          - Numeric path:     median imputation → standard scaling
          - Categorical path: most-frequent imputation → ordinal encoding

        OrdinalEncoder is set to map unseen categories to -1 at inference time
        so the predict pipeline never raises on new categorical values.
        """
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])

        return ColumnTransformer([
            ("num", numeric_pipeline,      numeric_cols),
            ("cat", categorical_pipeline,  categorical_cols),
        ])

    def run(
        self,
        train_path: str,
        test_path: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw train/test CSVs, apply feature engineering, fit the
        preprocessor on the training set, and transform both sets.

        The preprocessor is fitted only on training data to prevent any
        test-set statistics from leaking into the scaling or encoding steps.

        Returns
        -------
        (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting data transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info(f"Loaded train {train_df.shape} and test {test_df.shape}")

            train_df = apply_feature_engineering(train_df)
            test_df  = apply_feature_engineering(test_df)

            X_train = train_df.drop(columns=[TARGET])
            y_train = train_df[TARGET].values

            X_test = test_df.drop(columns=[TARGET])
            y_test = test_df[TARGET].values

            numeric_cols     = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
            logger.info(f"Numeric columns: {numeric_cols}")
            logger.info(f"Categorical columns: {categorical_cols}")

            preprocessor = self._build_preprocessor(numeric_cols, categorical_cols)

            # Fit on train only — transform is all that happens to test
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr  = preprocessor.transform(X_test)

            save_object(self.config.preprocessor_path, preprocessor)
            logger.info(f"Preprocessor saved to {self.config.preprocessor_path}")

            return X_train_arr, X_test_arr, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

"""
predict_pipeline.py
-------------------
Loads the saved preprocessor and model, applies identical feature engineering,
and returns a profit prediction for a single order.

PredictInput accepts the raw field values a user would enter through the
Streamlit UI or an API call, and converts them to a one-row DataFrame that
mirrors the training feature space exactly.
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.components.data_transformation import apply_feature_engineering, TARGET
from src.exception import CustomException
from src.logger import logger
from src.utils import load_object

PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
MODEL_PATH        = os.path.join("artifacts", "model.pkl")


@dataclass
class PredictInput:
    """
    Raw field values for a single FMCG order.

    All fields map 1-to-1 to columns in the raw dataset. The pipeline handles
    feature engineering and preprocessing internally.
    """
    year:                int
    quarter:             str    # e.g. "Q1", "Q2", "Q3", "Q4"
    month:               int
    month_name:          str    # e.g. "January"
    region:              str
    country:             str
    city:                str
    sales_person:        str
    customer_type:       str    # "B2B" or "B2C"
    sales_channel:       str    # "Modern Trade", "Online", "Distributor", "Wholesale"
    promotion_type:      str    # "No Promo", "Seasonal Campaign", ...
    product_category:    str
    brand:               str
    product_name:        str
    units_sold:          int
    unit_price_usd:      float
    discount_pct:        float
    marketing_spend_usd: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the input dataclass to a one-row DataFrame with raw column names."""
        return pd.DataFrame([{
            "Year":                self.year,
            "Quarter":             self.quarter,
            "Month":               self.month,
            "Month_Name":          self.month_name,
            "Region":              self.region,
            "Country":             self.country,
            "City":                self.city,
            "Sales_Person":        self.sales_person,
            "Customer_Type":       self.customer_type,
            "Sales_Channel":       self.sales_channel,
            "Promotion_Type":      self.promotion_type,
            "Product_Category":    self.product_category,
            "Brand":               self.brand,
            "Product_Name":        self.product_name,
            "Units_Sold":          self.units_sold,
            "Unit_Price_USD":      self.unit_price_usd,
            "Discount_Pct":        self.discount_pct,
            "Marketing_Spend_USD": self.marketing_spend_usd,
        }])


class PredictPipeline:
    def __init__(self):
        self.preprocessor = load_object(PREPROCESSOR_PATH)
        self.model        = load_object(MODEL_PATH)
        logger.info("PredictPipeline initialised")

    def predict(self, input_data: PredictInput) -> float:
        """
        Apply feature engineering → preprocessing → model inference.

        Feature engineering here must exactly mirror apply_feature_engineering()
        in data_transformation.py. The target column is never present at
        inference time, so we call apply_feature_engineering on a raw row that
        has no TARGET column — this is safe because TARGET is only separated,
        not required, by that function.

        Returns
        -------
        Predicted Profit_USD as a float.
        """
        try:
            raw_df = input_data.to_dataframe()

            # apply_feature_engineering drops LEAKAGE_COLS, which are absent
            # from raw_df at inference time — that's expected and fine.
            fe_df = apply_feature_engineering(raw_df)

            # Remove TARGET if somehow present (it won't be at inference time)
            if TARGET in fe_df.columns:
                fe_df = fe_df.drop(columns=[TARGET])

            X_arr = self.preprocessor.transform(fe_df)
            prediction = float(self.model.predict(X_arr)[0])

            logger.info(f"Prediction: {prediction:.2f}")
            return prediction

        except Exception as e:
            raise CustomException(e, sys)

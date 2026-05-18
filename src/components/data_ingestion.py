import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    raw_data_path:   str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path:  str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def run(self, source_path: str) -> tuple[str, str]:
        """
        Read the raw dataset from `source_path`, save a copy to artifacts/,
        then produce an 80/20 train/test split.

        Returns
        -------
        (train_path, test_path)
        """
        logger.info("Starting data ingestion")
        try:
            df = pd.read_csv(source_path)
            logger.info(f"Dataset loaded: {df.shape} rows × columns")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved to {self.config.raw_data_path}")

            train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info(
                f"Split complete — train: {train_df.shape}, test: {test_df.shape}"
            )

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

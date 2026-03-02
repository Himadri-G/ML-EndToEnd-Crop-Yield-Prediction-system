import os
from pathlib import Path
import joblib

import pandas as pd
from sklearn.preprocessing import StandardScaler

from crop_yield_prediction.entity.config_entity import DataPreprocessingConfig
from crop_yield_prediction.utils.logger import get_logger

logger = get_logger(
    name=__name__,
    log_file="data_preprocessing.log"
)

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler()

    def _load_data(self):
        logger.info("Loading train and test data")
        train_df = pd.read_csv(self.config.train_dir)
        test_df = pd.read_csv(self.config.test_dir)

        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")

        return train_df, test_df

    def _split_features_target(self, df):
        X = df.drop(columns=["label"])
        y = df["label"]
        return X, y

    def main_data_preprocessing(self):
        
            logger.info("Starting Data Preprocessing !")

            train_df, test_df = self._load_data()

            X_train, y_train = self._split_features_target(train_df)
            X_test, y_test = self._split_features_target(test_df)

            logger.info("Applying StandardScaler")

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            train_processed["label"] = y_train.values

            test_processed = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            test_processed["label"] = y_test.values

            Path(self.config.processed_train_dir).parent.mkdir(parents=True, exist_ok=True)

            train_processed.to_csv(self.config.processed_train_dir, index=False)
            test_processed.to_csv(self.config.processed_test_dir, index=False)

            joblib.dump(self.scaler, self.config.scaler_path)

            logger.info("Data Preprocessing Completed !")

            return self.config.processed_train_dir, self.config.processed_test_dir

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from crop_yield_prediction.entity.config_entity import DataIngestionConfig
from crop_yield_prediction.utils.logger import get_logger


logger = get_logger(
    name=__name__,
    log_file="data_ingestion.log"
)


class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def _read_data(self):
        try:
            logger.info("Reading data from source")
            
            df = pd.read_csv(self.config.source_dir)

            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
                logger.info("Dropped 'Unnamed: 0' column")

            return df
        
        except Exception as e:
            logger.error(f"Error while reading data: {e}")
            raise e
    
    
    def _save_rawData(self, df):
        try:
            raw_data_path = Path(self.config.root_dir) / "raw.csv"
            
            # ✅ Save without index to prevent future issues
            df.to_csv(raw_data_path, index=False)
            
            logger.info(f"Raw data saved at {raw_data_path}")
            return raw_data_path
        
        except Exception as e:
            logger.error(f"Error while saving raw data: {e}")
            raise e
    
    
    def _split_data(self, df):
        try:
            logger.info("Splitting data into train and test")
            
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )
            
            logger.info("Data split completed")
            return train_df, test_df
        
        except Exception as e:
            logger.error(f"Error while splitting data: {e}")
            raise e
    
    
    def main_DataIngestion_part(self):
        try:
            logger.info("Starting Data Ingestion process")
            
            # ✅ Create root directory
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # Step 1: Read data
            df = self._read_data()
            
            # Step 2: Save raw data
            self._save_rawData(df)
            
            # Step 3: Split data
            train_df, test_df = self._split_data(df)
            
            # Step 4: Save train & test data
            train_df.to_csv(self.config.train_dir, index=False)
            test_df.to_csv(self.config.test_dir, index=False)
            
            logger.info("Train and test data saved successfully")
            logger.info("Data Ingestion completed ✅")
            
            return self.config.train_dir, self.config.test_dir
        
        except Exception as e:
            logger.error(f"Error in Data Ingestion pipeline: {e}")
            raise e
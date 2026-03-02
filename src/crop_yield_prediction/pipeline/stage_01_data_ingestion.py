import os
from pathlib import Path
import logging

from crop_yield_prediction.components.data_ingestion import DataIngestion
from crop_yield_prediction.configuration.config import ConfigManager
from crop_yield_prediction.utils.logger import get_logger


logger = get_logger(
    name = __name__,
    file_name = "Pipeline.log"
    
)

STAGE_NAME = "Data Ingestion Stage"


def main():
    logger.info(f"{STAGE_NAME} is started...")
    
    config_manager = ConfigManager(
        file_path = Path("config/config.yaml")
    )
    
    data_ingestion_config = config_manager.get_data_config()
    
    data_ingestion = DataIngestion(config = data_ingestion_config)
    
    status = data_ingestion.main_DataIngestion_part()
    
    if not status:
        raise Exception(f"{STAGE_NAME} is failed")
    
    logger.info(f"{STAGE_NAME} is completed !")
    
if __name__ == "__main__":
    main()
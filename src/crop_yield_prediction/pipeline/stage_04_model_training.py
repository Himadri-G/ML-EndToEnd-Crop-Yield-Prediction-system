from pathlib import Path

from crop_yield_prediction.components.model_training import ModelTraining
from crop_yield_prediction.configuration.config import ConfigManager

from crop_yield_prediction.utils.logger import get_logger


logger = get_logger(
    name = __name__,
    file_name = "Pipeline.log"
    
)

STAGE_NAME = "Model Training Stage"


def main():
    logger.info(f"{STAGE_NAME} is started...")
    
    config_manager = ConfigManager(
        file_path = Path("config/config.yaml")
    )
    
    model_training_config = config_manager.get_data_config()
    
    model_training= ModelTraining(config = model_training_config)
    
    status = model_training.main_ModelTraining_part()
    
    if not status:
        raise Exception(f"{STAGE_NAME} is failed")
    
    logger.info(f"{STAGE_NAME} is completed !")
    
if __name__ == "__main__":
    main()
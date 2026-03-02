from pathlib import Path

from crop_yield_prediction.components.model_evalution import ModelEvaluation
from crop_yield_prediction.configuration.config import ConfigManager

from crop_yield_prediction.utils.logger import get_logger


logger = get_logger(
    name = __name__,
    file_name = "Pipeline.log"
    
)

STAGE_NAME = "Model Evaluation Stage"


def main():
    logger.info(f"{STAGE_NAME} is started...")
    
    config_manager = ConfigManager(
        file_path = Path("config/config.yaml")
    )
    
    model_evaluation_config = config_manager.get_data_config()
    
    model_evaluation= ModelEvaluation(config = model_evaluation_config)
    
    status = model_evaluation.main_ModelEvaluation_part()
    
    if not status:
        raise Exception(f"{STAGE_NAME} is failed")
    
    logger.info(f"{STAGE_NAME} is completed !")
    
if __name__ == "__main__":
    main()
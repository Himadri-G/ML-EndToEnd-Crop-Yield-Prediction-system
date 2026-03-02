from pathlib import Path
import os
import yaml

from crop_yield_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataPreprocessingConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig
)


class ConfigManager:

    def __init__(self, config_path: Path):
        self.config = self._read_yaml(config_path)

    def _read_yaml(self, file_path: Path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        ingestion = self.config["data_ingestion"]
        os.makedirs(ingestion["root_dir"], exist_ok=True)

        return DataIngestionConfig(
            root_dir=Path(ingestion["root_dir"]),
            source_path=Path(ingestion["source_path"]),
            train_path=Path(ingestion["train_path"]),
            test_path=Path(ingestion["test_path"])
        )

 
    def get_data_validation_config(self) -> DataValidationConfig:

        validation = self.config["data_validation"]
        os.makedirs(validation["root_dir"], exist_ok=True)

        return DataValidationConfig(
            root_dir=Path(validation["root_dir"]),
            validation_status_file=Path(validation["validation_status_file"]),
            train_path=Path(validation["train_path"]),
            schema_file=Path(validation["schema_file"])
        )

  
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        preprocessing = self.config["data_preprocessing"]
        ingestion = self.config["data_ingestion"]

        os.makedirs(preprocessing["root_dir"], exist_ok=True)

        return DataPreprocessingConfig(
            root_dir=Path(preprocessing["root_dir"]),
            train_path=Path(ingestion["train_path"]),
            test_path=Path(ingestion["test_path"]),
            processed_train_path=Path(preprocessing["processed_train_path"]),
            processed_test_path=Path(preprocessing["processed_test_path"]),
            scaler_path=Path(preprocessing["scaler_path"])
        )


    def get_model_trainer_config(self) -> ModelTrainingConfig:

        trainer = self.config["model_trainer"]
        preprocessing = self.config["data_preprocessing"]

        os.makedirs(trainer["root_dir"], exist_ok=True)

        return ModelTrainingConfig(
            root_dir=Path(trainer["root_dir"]),
            processed_train_path=Path(preprocessing["processed_train_path"]),
            processed_test_path=Path(preprocessing["processed_test_path"]),
            model_path=Path(trainer["model_path"]),
            params_file=Path(trainer["params_file"])
        )

 
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        evaluation = self.config["model_evaluation"]

        os.makedirs(evaluation["root_dir"], exist_ok=True)

        return ModelEvaluationConfig(
            root_dir=Path(evaluation["root_dir"]),
            model_path=Path(evaluation["model_path"]),
            processed_test_path=Path(evaluation["processed_test_path"]),
            metrics_file=Path(evaluation["metrics_file"])
        )
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir : Path
    source_dir : Path
    train_dir : Path
    test_dir : Path
    
@dataclass
class DataValidationConfig:
    root_dir : Path
    validation_status_file : Path
    train_dir : Path
    schema_file : Path
    
@dataclass
class DataPreprocessingConfig:
    root_dir : Path
    train_dir : Path
    test_dir : Path
    preprocessed_train_dir : Path
    preprocessed_test_dir : Path
    scaler_path : Path
    
@dataclass
class ModelTrainingConfig:
    root_dir: Path
    preprocessed_train_dir: Path
    preprocessed_test_dir: Path
    model_path: Path
    params_file: Path
    target_column: str = "hg/ha_yield"  
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"  
    mlflow_experiment_name: str = "Crop Yield Prediction" 
    mlflow_registered_model_name: str = "gradiant boosting"
    
@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    preprocessed_test_dir: Path
    metrics_file: Path
    target_column: str = "hg/ha_yield"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"  
    mlflow_experiment_name: str = "Crop Yield Prediction" 
    mlflow_registered_model_name: str = "gradiant boosting" 
import pandas as pd
import yaml
import joblib
import optuna
import mlflow
import mlflow.sklearn
import numpy as np

from pathlib import Path

<<<<<<< HEAD
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

=======
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
>>>>>>> cbad126 (updated file)
from sklearn.metrics import r2_score, mean_squared_error

from crop_yield_prediction.entity.config_entity import ModelTrainingConfig
from crop_yield_prediction.utils.logger import get_logger

logger = get_logger(__name__, "model_training.log")


class ModelTraining:

    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    # ---------------- LOAD PARAMS ---------------- #

    def _load_params(self):

        with open(self.config.params_file, "r") as f:
            params = yaml.safe_load(f)

        if not params:
            raise ValueError("params.yaml is empty")

        if "optuna" not in params:
            raise KeyError("Missing optuna section in params.yaml")

        return params

    # ---------------- LOAD DATA ---------------- #

    def _load_data(self):

        train_df = pd.read_csv(self.config.preprocessed_train_dir)
        test_df = pd.read_csv(self.config.preprocessed_test_dir)

        return train_df, test_df

    # ---------------- SPLIT FEATURES ---------------- #

    def _split_features_target(self, df):

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        return X, y

    # ---------------- MODEL FACTORY ---------------- #

    def _get_model(self, model_name, params):

        if model_name == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 3)
            )

<<<<<<< HEAD
        
=======
>>>>>>> cbad126 (updated file)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # ---------------- OPTUNA OBJECTIVE ---------------- #

    def _objective(self, trial, model_name, param_space, X_train, y_train, X_test, y_test):

        params = {}

        for param, values in param_space.items():

            if values["type"] == "int":
                params[param] = trial.suggest_int(param, values["low"], values["high"])

            elif values["type"] == "float":
                params[param] = trial.suggest_float(param, values["low"], values["high"])

            elif values["type"] == "categorical":
                params[param] = trial.suggest_categorical(param, values["choices"])

        try:

            model = self._get_model(model_name, params)

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            mlflow.log_params(params)
            mlflow.log_metric(f"{model_name}_r2", r2)
            mlflow.log_metric(f"{model_name}_rmse", rmse)

            return r2

        except Exception as e:

            logger.error(f"Trial failed for {model_name}: {e}")
            return -1

    # ---------------- MAIN TRAINING ---------------- #

    def main_ModelTraining_part(self):

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        logger.info("Starting Model Training")

        params = self._load_params()

        n_trials = params["optuna"]["n_trials"]
        models_params = params["models"]

        train_df, test_df = self._load_data()

        X_train, y_train = self._split_features_target(train_df)
        X_test, y_test = self._split_features_target(test_df)

        best_model = None
        best_score = -np.inf
        best_model_name = None

        with mlflow.start_run(run_name="Regression_Model_Training"):

            for model_name, param_space in models_params.items():

                logger.info(f"Tuning model: {model_name}")

                study = optuna.create_study(direction="maximize")

                study.optimize(
                    lambda trial: self._objective(
                        trial,
                        model_name,
                        param_space,
                        X_train,
                        y_train,
                        X_test,
                        y_test
                    ),
                    n_trials=n_trials
                )

                best_params = study.best_params

                model = self._get_model(model_name, best_params)

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                logger.info(f"{model_name} R2 Score: {r2}")

                if r2 > best_score:

                    best_score = r2
                    best_model = model
                    best_model_name = model_name

<<<<<<< HEAD
        # ---------------- SAVE BEST MODEL ---------------- #

        Path(self.config.model_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_model, self.config.model_path)

        mlflow.sklearn.log_model(
            best_model,
=======
            logger.info(f"best model selected : {best_model_name}")

            # ---------------- CREATE PIPELINE ---------------- #

            pipeline = Pipeline([
                ("model", best_model)
            ])

        # ---------------- SAVE PIPELINE ---------------- #

        Path(self.config.model_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, self.config.model_path)

        # ---------------- LOG TO MLFLOW ---------------- #

        mlflow.sklearn.log_model(
            pipeline,
>>>>>>> cbad126 (updated file)
            artifact_path="model",
            registered_model_name=self.config.mlflow_registered_model_name
        )

        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best R2 Score: {best_score}")
<<<<<<< HEAD
=======
        logger.info(f"Pipeline saved at: {self.config.model_path}")
>>>>>>> cbad126 (updated file)

        return self.config.model_path
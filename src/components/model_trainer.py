import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ( 
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logger


class ModelTrainer:
    """
    Train and evaluate baseline demand prediction models.
    """

    def __init__(self):

        self.processed_directory = os.path.join(
            "artifacts",
            "processed"
        )

        self.models_directory = "models"

        self.results_path = os.path.join(
            self.processed_directory,
            "baseline_model_results.csv"
        )

        self.final_model_path = os.path.join(
            self.models_directory,
            "final_demand_model.pkl"
        )

    def _load_processed_data(self):

        logger.info(
            "Loading processed training and testing data."
        )

        X_train = pd.read_csv(
            os.path.join(
                self.processed_directory,
                "X_train.csv"
            )
        )

        X_test = pd.read_csv(
            os.path.join(
                self.processed_directory,
                "X_test.csv"
            )
        )

        y_train = pd.read_csv(
            os.path.join(
                self.processed_directory,
                "y_train.csv"
            )
        ).squeeze()

        y_test = pd.read_csv(
            os.path.join(
                self.processed_directory,
                "y_test.csv"
            )
        ).squeeze()

        logger.info(
            f"X_train shape: {X_train.shape}"
        )

        logger.info(
            f"X_test shape: {X_test.shape}"
        )

        logger.info(
            f"y_train shape: {y_train.shape}"
        )

        logger.info(
            f"y_test shape: {y_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    def _create_models(self):

        logger.info(
            "Creating baseline demand prediction models."
        )

        models = {

            "Linear Regression": Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]),

            "Decision Tree": DecisionTreeRegressor(
                random_state=42
            ),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            ),

            "XGBoost": XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
                n_jobs=-1
            ),

            "SVR": Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    SVR(
                        kernel="rbf",
                        C=10,
                        epsilon=0.1
                    )
                )
            ])
        }

        return models

    @staticmethod
    def _evaluate_model(model, X, y):

        predictions = model.predict(X)

        mae = mean_absolute_error(
            y,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y,
                predictions
            )
        )

        r2 = r2_score(
            y,
            predictions
        )

        return mae, rmse, r2

    def initiate_model_training(self):

        logger.info(
            "Starting Phase 6 model training."
        )

        try:

            # ---------------------------------------------
            # Load processed data
            # ---------------------------------------------

            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = self._load_processed_data()

            # ---------------------------------------------
            # Create models
            # ---------------------------------------------

            models = self._create_models()

            trained_models = {}
            results = []

            # ---------------------------------------------
            # Train and evaluate all models
            # ---------------------------------------------

            for name, model in models.items():

                logger.info(
                    f"Training {name}."
                )

                model.fit(
                    X_train,
                    y_train
                )

                trained_models[name] = model

                mae, rmse, r2 = (
                    self._evaluate_model(
                        model,
                        X_test,
                        y_test
                    )
                )

                results.append(
                    [
                        name,
                        mae,
                        rmse,
                        r2
                    ]
                )

                logger.info(
                    f"{name} | "
                    f"MAE={mae:.4f}, "
                    f"RMSE={rmse:.4f}, "
                    f"R2={r2:.4f}"
                )

            # ---------------------------------------------
            # Create comparison table
            # ---------------------------------------------

            results_df = pd.DataFrame(
                results,
                columns=[
                    "Model",
                    "MAE",
                    "RMSE",
                    "R2"
                ]
            )

            results_df = (
                results_df
                .sort_values(
                    by="RMSE"
                )
                .reset_index(drop=True)
            )

            # ---------------------------------------------
            # Save results
            # ---------------------------------------------

            os.makedirs(
                self.processed_directory,
                exist_ok=True
            )

            results_df.to_csv(
                self.results_path,
                index=False
            )

            # ---------------------------------------------
            # Select best baseline model
            # ---------------------------------------------

            best_model_name = (
                results_df.iloc[0]["Model"]
            )

            best_model = trained_models[
                best_model_name
            ]

            logger.info(
                f"Best baseline model: "
                f"{best_model_name}"
            )

            # ---------------------------------------------
            # Save baseline model
            # ---------------------------------------------

            os.makedirs(
                self.models_directory,
                exist_ok=True
            )

            joblib.dump(
                best_model,
                self.final_model_path
            )

            logger.info(
                f"Baseline model saved to: "
                f"{self.final_model_path}"
            )

            return (
                results_df,
                best_model_name,
                self.final_model_path
            )

        except Exception as error:

            logger.error(
                "Error occurred during model training."
            )

            raise CustomException(
                error,
                sys
            )


if __name__ == "__main__":

    trainer = ModelTrainer()

    (
        results,
        best_model,
        model_path
    ) = trainer.initiate_model_training()

    print("\n" + "=" * 60)
    print("PHASE 6 — MODEL TRAINING COMPLETED")
    print("=" * 60)

    print("\nModel Comparison:")
    print(results.to_string(index=False))

    print(
        f"\nBest Baseline Model: {best_model}"
    )

    print(
        f"Model Saved At: {model_path}"
    )


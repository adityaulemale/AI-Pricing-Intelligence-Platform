import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.exception import CustomException
from src.logger import logger


class ModelEvaluation:
    """
    Evaluate and tune the baseline Random Forest model
    using time-series cross-validation.
    """

    def __init__(self):

        self.processed_directory = os.path.join(
            "artifacts",
            "processed"
        )

        self.models_directory = "models"

        self.baseline_results_path = os.path.join(
            self.processed_directory,
            "baseline_model_results.csv"
        )

        self.evaluation_results_path = os.path.join(
            self.processed_directory,
            "model_evaluation_results.csv"
        )

        self.tuning_results_path = os.path.join(
            self.processed_directory,
            "rf_tuning_results.csv"
        )

        self.final_model_path = os.path.join(
            self.models_directory,
            "final_demand_model.pkl"
        )

    def _load_data(self):

        logger.info(
            "Loading processed data for model evaluation."
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

        return X_train, X_test, y_train, y_test

    @staticmethod
    def _calculate_metrics(y_true, predictions):

        mae = mean_absolute_error(
            y_true,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                predictions
            )
        )

        r2 = r2_score(
            y_true,
            predictions
        )

        return mae, rmse, r2

    def _get_baseline_random_forest_metrics(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        logger.info(
            "Training baseline Random Forest for comparison."
        )

        baseline_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        baseline_model.fit(
            X_train,
            y_train
        )

        baseline_predictions = (
            baseline_model.predict(X_test)
        )

        (
            baseline_mae,
            baseline_rmse,
            baseline_r2
        ) = self._calculate_metrics(
            y_test,
            baseline_predictions
        )

        return {
            "Model": "Original Random Forest",
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "R2": baseline_r2
        }

    def _tune_random_forest(
        self,
        X_train,
        y_train
    ):

        logger.info(
            "Starting Random Forest hyperparameter tuning."
        )

        # Same parameter grid as 04_Model_Training.ipynb
        rf_param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [
                None,
                10,
                15,
                20
            ],
            "min_samples_split": [
                2,
                5,
                10
            ],
            "min_samples_leaf": [
                1,
                2,
                4
            ],
            "max_features": [
                "sqrt",
                0.8
            ]
        }

        rf_tuning_model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )

        tscv = TimeSeriesSplit(
            n_splits=3
        )

        rf_search = RandomizedSearchCV(
            estimator=rf_tuning_model,
            param_distributions=rf_param_grid,
            n_iter=15,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        rf_search.fit(
            X_train,
            y_train
        )

        logger.info(
            f"Best parameters: "
            f"{rf_search.best_params_}"
        )

        logger.info(
            f"Best CV RMSE: "
            f"{-rf_search.best_score_}"
        )

        return rf_search

    def initiate_model_evaluation(self):

        logger.info(
            "Starting Phase 7 — Model Evaluation & Tuning."
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
            ) = self._load_data()

            logger.info(
                f"X_train shape: {X_train.shape}"
            )

            logger.info(
                f"X_test shape: {X_test.shape}"
            )

            # ---------------------------------------------
            # Baseline Random Forest
            # ---------------------------------------------

            baseline_result = (
                self._get_baseline_random_forest_metrics(
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )
            )

            logger.info(
                "Baseline Random Forest evaluation completed."
            )

            # ---------------------------------------------
            # Tune Random Forest
            # ---------------------------------------------

            rf_search = self._tune_random_forest(
                X_train,
                y_train
            )

            tuned_rf = rf_search.best_estimator_

            # ---------------------------------------------
            # Tuned model test predictions
            # ---------------------------------------------

            tuned_predictions = tuned_rf.predict(
                X_test
            )

            (
                tuned_mae,
                tuned_rmse,
                tuned_r2
            ) = self._calculate_metrics(
                y_test,
                tuned_predictions
            )

            tuned_result = {
                "Model": "Tuned Random Forest",
                "MAE": tuned_mae,
                "RMSE": tuned_rmse,
                "R2": tuned_r2
            }

            logger.info(
                f"Tuned Random Forest | "
                f"MAE={tuned_mae:.4f}, "
                f"RMSE={tuned_rmse:.4f}, "
                f"R2={tuned_r2:.4f}"
            )

            # ---------------------------------------------
            # Compare baseline and tuned models
            # ---------------------------------------------

            comparison_df = pd.DataFrame(
                [
                    baseline_result,
                    tuned_result
                ]
            )

            comparison_df = comparison_df.sort_values(
                by="RMSE"
            ).reset_index(drop=True)

            # ---------------------------------------------
            # Save evaluation results
            # ---------------------------------------------

            os.makedirs(
                self.processed_directory,
                exist_ok=True
            )

            comparison_df.to_csv(
                self.evaluation_results_path,
                index=False
            )

            # ---------------------------------------------
            # Save RandomizedSearchCV results
            # ---------------------------------------------

            cv_results = pd.DataFrame(
                rf_search.cv_results_
            )

            cv_results.to_csv(
                self.tuning_results_path,
                index=False
            )

            # ---------------------------------------------
            # Select final model
            # ---------------------------------------------

            best_model_name = (
                comparison_df.iloc[0]["Model"]
            )

            if best_model_name == "Tuned Random Forest":

                final_model = tuned_rf

                logger.info(
                    "Tuned Random Forest selected as "
                    "the final production model."
                )

            else:

                final_model = (
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    )
                )

                final_model.fit(
                    X_train,
                    y_train
                )

                logger.info(
                    "Original Random Forest retained as "
                    "the final production model."
                )

            # ---------------------------------------------
            # Save final production model
            # ---------------------------------------------

            os.makedirs(
                self.models_directory,
                exist_ok=True
            )

            joblib.dump(
                final_model,
                self.final_model_path
            )

            logger.info(
                f"Final model saved at: "
                f"{self.final_model_path}"
            )

            return (
                comparison_df,
                best_model_name,
                self.final_model_path
            )

        except Exception as error:

            logger.error(
                "Error occurred during model evaluation "
                "and tuning."
            )

            raise CustomException(
                error,
                sys
            )


if __name__ == "__main__":

    evaluator = ModelEvaluation()

    (
        results,
        best_model,
        model_path
    ) = evaluator.initiate_model_evaluation()

    print("\n" + "=" * 65)
    print("PHASE 7 — MODEL EVALUATION & TUNING COMPLETED")
    print("=" * 65)

    print("\nBaseline vs Tuned Performance:")
    print(results.to_string(index=False))

    print(
        f"\nSelected Final Model: {best_model}"
    )

    print(
        f"Final Model Saved At: {model_path}"
    )


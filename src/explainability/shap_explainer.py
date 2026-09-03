from src.pipeline.predict_pipeline import (
    PredictPipeline
)

import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

from src.exception import CustomException
from src.logger import logger


class SHAPExplainer:
    """
    SHAP explainability engine for the final demand model.

    Uses SHAP TreeExplainer for the trained Random Forest
    model and the same fitted preprocessing artifacts used
    by the prediction pipeline.
    """

    def __init__(self):

        self.preprocessor_path = os.path.join(
            "artifacts",
            "processed",
            "preprocessor.pkl"
        )

        self.feature_config_path = os.path.join(
            "artifacts",
            "processed",
            "feature_config.pkl"
        )

        self.model_path = os.path.join(
            "models",
            "final_demand_model.pkl"
        )

        self.preprocessor = None
        self.feature_config = None
        self.model = None
        self.explainer = None

    # =========================================================
    # LOAD ARTIFACTS
    # =========================================================

    def _load_artifacts(self):
        """
        Load the fitted preprocessor, feature configuration,
        and final trained model.
        """

        logger.info(
            "Loading SHAP artifacts."
        )

        if not os.path.exists(
            self.preprocessor_path
        ):
            raise FileNotFoundError(
                f"Preprocessor not found: "
                f"{self.preprocessor_path}"
            )

        if not os.path.exists(
            self.feature_config_path
        ):
            raise FileNotFoundError(
                f"Feature config not found: "
                f"{self.feature_config_path}"
            )

        if not os.path.exists(
            self.model_path
        ):
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

        self.feature_config = joblib.load(
            self.feature_config_path
        )

        self.model = joblib.load(
            self.model_path
        )

        logger.info(
            "SHAP artifacts loaded successfully."
        )

    # =========================================================
    # PREPARE FEATURES
    # =========================================================

    def _prepare_features(
        self,
        transformed_input
    ):
        """
        Apply the fitted preprocessor and return a DataFrame
        containing transformed features with feature names.
        """

        logger.info(
            "Preparing features for SHAP analysis."
        )

        processed_input = (
            self.preprocessor.transform(
                transformed_input
            )
        )

        # -----------------------------------------------------
        # Get transformed feature names
        # -----------------------------------------------------

        if hasattr(
            self.preprocessor,
            "get_feature_names_out"
        ):

            feature_names = (
                self.preprocessor
                .get_feature_names_out()
            )

        else:

            feature_names = [
                f"feature_{index}"
                for index in range(
                    processed_input.shape[1]
                )
            ]

        # -----------------------------------------------------
        # Convert transformed data into DataFrame
        # -----------------------------------------------------

        processed_input = pd.DataFrame(
            processed_input,
            columns=feature_names,
            index=transformed_input.index
        )

        return processed_input

    # =========================================================
    # CREATE SHAP EXPLAINER
    # =========================================================

    def _create_explainer(self):
        """
        Create SHAP TreeExplainer for the final Random Forest.
        """

        logger.info(
            "Creating SHAP TreeExplainer."
        )

        self.explainer = (
            shap.TreeExplainer(
                self.model
            )
        )

        logger.info(
            "SHAP TreeExplainer created successfully."
        )

    # =========================================================
    # EXPLAIN PREDICTION
    # =========================================================

    def explain(
        self,
        transformed_input
    ):
        """
        Generate SHAP explanations.

        Parameters
        ----------
        transformed_input : pandas.DataFrame
            DataFrame after the same feature engineering
            used during model training and prediction.

        Returns
        -------
        dict
            SHAP explanation results.
        """

        logger.info(
            "Starting SHAP explanation."
        )

        try:

            # -------------------------------------------------
            # Validate input
            # -------------------------------------------------

            if not isinstance(
                transformed_input,
                pd.DataFrame
            ):
                raise TypeError(
                    "transformed_input must be "
                    "a pandas DataFrame."
                )

            if transformed_input.empty:
                raise ValueError(
                    "SHAP input is empty."
                )

            # -------------------------------------------------
            # Load artifacts
            # -------------------------------------------------

            self._load_artifacts()

            # -------------------------------------------------
            # Prepare processed features
            # -------------------------------------------------

            processed_input = (
                self._prepare_features(
                    transformed_input
                )
            )

            # -------------------------------------------------
            # Create SHAP explainer
            # -------------------------------------------------

            self._create_explainer()

            # -------------------------------------------------
            # Calculate SHAP values
            # -------------------------------------------------

            shap_values = (
                self.explainer.shap_values(
                    processed_input
                )
            )

            shap_values = np.asarray(
                shap_values
            )

            # -------------------------------------------------
            # Handle possible SHAP output dimensions
            # -------------------------------------------------

            if shap_values.ndim == 3:

                shap_values = (
                    shap_values[:, :, 0]
                )

            if shap_values.ndim != 2:

                raise ValueError(
                    "Unexpected SHAP value shape: "
                    f"{shap_values.shape}"
                )

            # -------------------------------------------------
            # Expected / base value
            # -------------------------------------------------

            expected_value = (
                self.explainer.expected_value
            )

            expected_value = np.asarray(
                expected_value
            )

            base_value = float(
                expected_value.reshape(-1)[0]
            )

            # -------------------------------------------------
            # Model predictions
            # -------------------------------------------------

            predictions = (
                self.model.predict(
                    processed_input
                )
            )

            predictions = np.asarray(
                predictions
            )

            # -------------------------------------------------
            # Global feature importance
            # -------------------------------------------------

            mean_abs_shap = (
                np.abs(
                    shap_values
                ).mean(axis=0)
            )

            feature_importance = (
                pd.DataFrame(
                    {
                        "feature":
                            processed_input.columns,

                        "mean_abs_shap":
                            mean_abs_shap
                    }
                )
                .sort_values(
                    "mean_abs_shap",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            # -------------------------------------------------
            # Individual explanations
            # -------------------------------------------------

            individual_explanations = []

            for row_index in range(
                len(processed_input)
            ):

                row_explanation = (
                    pd.DataFrame(
                        {
                            "feature":
                                processed_input.columns,

                            "feature_value":
                                processed_input
                                .iloc[row_index]
                                .values,

                            "shap_value":
                                shap_values[
                                    row_index
                                ]
                        }
                    )
                )

                row_explanation[
                    "absolute_shap_value"
                ] = np.abs(
                    row_explanation[
                        "shap_value"
                    ]
                )

                row_explanation = (
                    row_explanation
                    .sort_values(
                        "absolute_shap_value",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )

                individual_explanations.append(
                    row_explanation
                )

            # -------------------------------------------------
            # SHAP prediction reconstruction
            # -------------------------------------------------

            reconstructed_predictions = (
                base_value
                + shap_values.sum(axis=1)
            )

            # -------------------------------------------------
            # Explanation consistency
            # -------------------------------------------------

            prediction_difference = (
                predictions
                - reconstructed_predictions
            )

            logger.info(
                "SHAP explanation completed successfully."
            )

            return {

                "predictions":
                    predictions,

                "base_value":
                    base_value,

                "shap_values":
                    shap_values,

                "feature_values":
                    processed_input,

                "feature_importance":
                    feature_importance,

                "individual_explanations":
                    individual_explanations,

                "reconstructed_predictions":
                    reconstructed_predictions,

                "prediction_difference":
                    prediction_difference
            }

        except Exception as error:

            logger.error(
                "Error occurred during SHAP explanation."
            )

            raise CustomException(
                error,
                sys
            )
        
    def generate_plots(
        self,
        result,
        output_directory="artifacts/explainability"
    ):
        """
        Generate global and local SHAP visualizations.
        """

        logger.info(
            "Generating SHAP visualizations."
        )

        try:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

            # --------------------------------------------------
            # Global feature importance
            # --------------------------------------------------

            global_importance = (
                result["feature_importance"]
                .head(15)
                .copy()
            )

            global_plot_path = os.path.join(
                output_directory,
                "shap_feature_importance.png"
            )

            import matplotlib.pyplot as plt

            plt.figure(
                figsize=(10, 7)
            )

            plt.barh(
                global_importance["feature"][::-1],
                global_importance[
                    "mean_abs_shap"
                ][::-1]
            )

            plt.xlabel(
                "Mean Absolute SHAP Value"
            )

            plt.ylabel(
                "Feature"
            )

            plt.title(
                "Global SHAP Feature Importance"
            )

            plt.tight_layout()

            plt.savefig(
                global_plot_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            # --------------------------------------------------
            # Local feature contribution
            # --------------------------------------------------

            local_explanation = (
                result[
                    "individual_explanations"
                ][0]
                .head(15)
                .copy()
            )

            local_plot_path = os.path.join(
                output_directory,
                "shap_prediction_explanation.png"
            )

            plt.figure(
                figsize=(10, 7)
            )

            plt.barh(
                local_explanation["feature"][::-1],
                local_explanation[
                    "shap_value"
                ][::-1]
            )

            plt.axvline(
                x=0,
                linestyle="--"
            )

            plt.xlabel(
                "SHAP Value"
            )

            plt.ylabel(
                "Feature"
            )

            plt.title(
                "SHAP Explanation for Prediction"
            )

            plt.tight_layout()

            plt.savefig(
                local_plot_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                "SHAP visualizations generated successfully."
            )

            return {
                "global_plot":
                    global_plot_path,

                "local_plot":
                    local_plot_path
            }

        except Exception as error:

            logger.error(
                "Error occurred while generating "
                "SHAP visualizations."
            )

            raise CustomException(
                error,
                sys
            )



    def explain_prediction(
        self,
        input_data
    ):
        """
        Generate a SHAP explanation directly from
        raw prediction input.

        Uses the same feature engineering logic as
        PredictPipeline.
        """

        try:

            if isinstance(
                input_data,
                dict
            ):

                input_dataframe = pd.DataFrame(
                    [input_data]
                )

            elif isinstance(
                input_data,
                pd.Series
            ):

                input_dataframe = (
                    input_data.to_frame().T
                )

            elif isinstance(
                input_data,
                pd.DataFrame
            ):

                input_dataframe = (
                    input_data.copy()
                )

            else:

                raise TypeError(
                    "input_data must be a "
                    "dictionary, Series, or DataFrame."
                )

            if input_dataframe.empty:
                raise ValueError(
                    "SHAP input is empty."
                )

            pipeline = PredictPipeline()

            feature_config_path = os.path.join(
                "artifacts",
                "processed",
                "feature_config.pkl"
            )

            if not os.path.exists(
                feature_config_path
            ):
                raise FileNotFoundError(
                    f"Feature configuration not found: "
                    f"{feature_config_path}"
                )

            feature_config = joblib.load(
                feature_config_path
            )

            transformed_input = (
                pipeline._feature_engineering(
                    input_dataframe,
                    feature_config
                )
            )

            return self.explain(
                transformed_input
            )

        except Exception as error:

            logger.error(
                "Error occurred while generating "
                "prediction explanation."
            )

            raise CustomException(
                error,
                sys
            )    


# =============================================================
# MAIN TEST
# =============================================================

if __name__ == "__main__":
    try:

        # -----------------------------------------------------
        # Create prediction pipeline
        # -----------------------------------------------------

        pipeline = PredictPipeline()

        # -----------------------------------------------------
        # Sample input
        # -----------------------------------------------------

        sample_input = {
            "date": "2023-09-01",
            "store_id": "S001",
            "product_id": "P001",
            "category": "Groceries",
            "region": "North",
            "inventory_level": 500,
            "price": 50.0,
            "discount": 10.0,
            "weather_condition": "Sunny",
            "promotion": 1,
            "competitor_pricing": 48.0,
            "seasonality": "Summer",
            "epidemic": 0
        }

        # -----------------------------------------------------
        # Load feature configuration
        # -----------------------------------------------------

        feature_config_path = os.path.join(
            "artifacts",
            "processed",
            "feature_config.pkl"
        )

        if not os.path.exists(
            feature_config_path
        ):
            raise FileNotFoundError(
                f"Feature config not found: "
                f"{feature_config_path}"
            )

        feature_config = joblib.load(
            feature_config_path
        )

        # -----------------------------------------------------
        # Apply EXACT same feature engineering
        # used by prediction pipeline
        # -----------------------------------------------------

        transformed_input = (
            pipeline._feature_engineering(
                pd.DataFrame(
                    [sample_input]
                ),
                feature_config
            )
        )

        # -----------------------------------------------------
        # Create SHAP explainer
        # -----------------------------------------------------

        explainer = SHAPExplainer()

        # -----------------------------------------------------
        # Generate explanation
        # -----------------------------------------------------

        result = explainer.explain(
            transformed_input
        )

        plot_paths = (
            explainer.generate_plots(
                result
            )
        )

        print(
            "\nSHAP plots generated:"
        )

        print(
            "Global:",
            plot_paths["global_plot"]
        )

        print(
            "Local:",
            plot_paths["local_plot"]
        )

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "PHASE 10 — SHAP EXPLAINABILITY"
        )

        print(
            "=" * 60
        )

        print(
            "\nPredicted Demand:",
            f"{result['predictions'][0]:.2f}"
        )

        print(
            "SHAP Base Value:",
            f"{result['base_value']:.2f}"
        )

        print(
            "\nSHAP Reconstructed Prediction:",
            f"{result['reconstructed_predictions'][0]:.2f}"
        )

        print(
            "Prediction Difference:",
            f"{result['prediction_difference'][0]:.10f}"
        )

        # -----------------------------------------------------
        # Global feature importance
        # -----------------------------------------------------

        print(
            "\nTop 10 Important Features:"
        )

        print(
            result[
                "feature_importance"
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        # -----------------------------------------------------
        # Individual explanation
        # -----------------------------------------------------

        print(
            "\nTop 10 Prediction Drivers:"
        )

        print(
            result[
                "individual_explanations"
            ][0]
            .head(10)
            .to_string(
                index=False
            )
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "SHAP TEST COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 60
        )

    except Exception as error:

        print(
            "\nSHAP test failed."
        )

        print(
            f"Error: {error}"
        )

        raise


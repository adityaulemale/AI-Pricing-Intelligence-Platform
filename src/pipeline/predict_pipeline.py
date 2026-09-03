import os
import sys

import joblib
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger


class PredictPipeline:
    """
    Production prediction pipeline.

    Loads:
        1. Trained preprocessor
        2. Training-time feature configuration
        3. Final demand prediction model

    Applies the same feature engineering logic used during
    training and generates demand predictions.
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

    def _load_artifacts(self):

        logger.info(
            "Loading prediction artifacts."
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
                f"Feature configuration not found: "
                f"{self.feature_config_path}"
            )

        if not os.path.exists(
            self.model_path
        ):
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        preprocessor = joblib.load(
            self.preprocessor_path
        )

        feature_config = joblib.load(
            self.feature_config_path
        )

        model = joblib.load(
            self.model_path
        )

        logger.info(
            "Preprocessor, feature configuration, "
            "and model loaded successfully."
        )

        return (
            preprocessor,
            feature_config,
            model
        )

    @staticmethod
    def _apply_quantile_bins(
        series,
        bins,
        feature_name
    ):
        """
        Apply the quantile boundaries learned during training.

        IMPORTANT:
        We do NOT calculate new quantiles during prediction.
        """

        if bins is None:
            raise ValueError(
                f"Training bins for {feature_name} "
                f"are missing."
            )

        bins = np.asarray(
            bins,
            dtype=float
        )

        if len(bins) < 2:
            raise ValueError(
                f"Invalid training bins for "
                f"{feature_name}."
            )

        interval_count = len(bins) - 1

        labels = [
            "Low",
            "Medium",
            "High"
        ][:interval_count]

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        result = pd.cut(
            numeric_series,
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # Values outside the training range are assigned
        # to the closest available level.
        result = result.astype(object)

        below_min = (
            numeric_series < bins[0]
        )

        above_max = (
            numeric_series > bins[-1]
        )

        if "Low" in labels:
            result.loc[below_min] = "Low"

        if "High" in labels:
            result.loc[above_max] = "High"

        # Any remaining missing values are assigned to Medium
        # when available.
        if "Medium" in labels:
            result = result.where(
                result.notna(),
                "Medium"
            )
        else:
            result = result.where(
                result.notna(),
                labels[0]
            )

        return result

    @classmethod
    def _feature_engineering(
        cls,
        dataframe,
        feature_config
    ):
        """
        Apply the same feature engineering logic used during
        training.

        Quantile-based features use boundaries learned from
        training data.
        """

        logger.info(
            "Applying prediction-time feature engineering."
        )

        dataframe = dataframe.copy()

        # --------------------------------------------------
        # Validate required columns
        # --------------------------------------------------

        required_columns = [
            "date",
            "store_id",
            "product_id",
            "category",
            "region",
            "inventory_level",
            "price",
            "discount",
            "weather_condition",
            "promotion",
            "seasonality",
            "epidemic"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required prediction columns: "
                f"{missing_columns}"
            )

        # --------------------------------------------------
        # Date
        # --------------------------------------------------

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce"
        )

        if dataframe["date"].isna().any():
            raise ValueError(
                "Invalid date supplied for prediction."
            )

        # --------------------------------------------------
        # Remove leakage columns if supplied
        # --------------------------------------------------

        dataframe = dataframe.drop(
            columns=[
                "units_sold",
                "units_ordered"
            ],
            errors="ignore"
        )

        # --------------------------------------------------
        # Date features
        # --------------------------------------------------

        dataframe["year"] = (
            dataframe["date"].dt.year
        )

        dataframe["month"] = (
            dataframe["date"].dt.month
        )

        dataframe["day"] = (
            dataframe["date"].dt.day
        )

        dataframe["day_of_week"] = (
            dataframe["date"].dt.dayofweek
        )

        dataframe["is_weekend"] = (
            dataframe["day_of_week"] >= 5
        ).astype(int)

        dataframe["week_of_year"] = (
            dataframe["date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        # --------------------------------------------------
        # Cyclical features
        # --------------------------------------------------

        dataframe["month_sin"] = np.sin(
            2
            * np.pi
            * dataframe["month"]
            / 12
        )

        dataframe["month_cos"] = np.cos(
            2
            * np.pi
            * dataframe["month"]
            / 12
        )

        dataframe["day_of_week_sin"] = np.sin(
            2
            * np.pi
            * dataframe["day_of_week"]
            / 7
        )

        dataframe["day_of_week_cos"] = np.cos(
            2
            * np.pi
            * dataframe["day_of_week"]
            / 7
        )

        # --------------------------------------------------
        # Competitive pricing
        # --------------------------------------------------

        if "competitor_pricing" in dataframe.columns:

            competitor_price = (
                dataframe["competitor_pricing"]
            )

            dataframe["price_difference"] = (
                dataframe["price"]
                - competitor_price
            )

            dataframe["price_premium_pct"] = (
                (
                    dataframe["price"]
                    - competitor_price
                )
                / competitor_price
            ) * 100

            dataframe["price_ratio"] = (
                dataframe["price"]
                / competitor_price
            )

            dataframe = dataframe.drop(
                columns=["competitor_pricing"],
                errors="ignore"
            )

        else:

            raise ValueError(
                "competitor_pricing is required "
                "for prediction."
            )

        # --------------------------------------------------
        # Price level
        #
        # Uses TRAINING quantile boundaries.
        # --------------------------------------------------

        dataframe["price_level"] = (
            cls._apply_quantile_bins(
                dataframe["price"],
                feature_config["price_bins"],
                "price"
            )
        )

        # --------------------------------------------------
        # Discount level
        #
        # Uses TRAINING quantile boundaries.
        # --------------------------------------------------

        dataframe["discount_level"] = (
            cls._apply_quantile_bins(
                dataframe["discount"],
                feature_config["discount_bins"],
                "discount"
            )
        )

        # --------------------------------------------------
        # Interaction features
        # --------------------------------------------------

        dataframe["promotion_discount"] = (
            dataframe["promotion"]
            * dataframe["discount"]
        )

        dataframe["price_promotion"] = (
            dataframe["price"]
            * dataframe["promotion"]
        )

        dataframe["price_discount"] = (
            dataframe["price"]
            * dataframe["discount"]
        )

        # --------------------------------------------------
        # Category interactions
        # --------------------------------------------------

        dataframe["category_seasonality"] = (
            dataframe["category"].astype(str)
            + "_"
            + dataframe["seasonality"].astype(str)
        )

        dataframe["category_price_level"] = (
            dataframe["category"].astype(str)
            + "_"
            + dataframe["price_level"].astype(str)
        )

        # --------------------------------------------------
        # Inventory transformation
        # --------------------------------------------------

        dataframe["inventory_log"] = np.log1p(
            dataframe["inventory_level"]
        )

        # --------------------------------------------------
        # Remove raw date
        # --------------------------------------------------

        dataframe = dataframe.drop(
            columns=["date"]
        )

        return dataframe

    def predict(self, input_data):

        logger.info(
            "Starting demand prediction."
        )

        try:

            # --------------------------------------------------
            # Convert input to DataFrame
            # --------------------------------------------------

            if isinstance(
                input_data,
                dict
            ):

                input_data = pd.DataFrame(
                    [input_data]
                )

            elif isinstance(
                input_data,
                pd.Series
            ):

                input_data = input_data.to_frame().T

            elif not isinstance(
                input_data,
                pd.DataFrame
            ):

                raise TypeError(
                    "input_data must be a "
                    "dictionary, Series, or DataFrame."
                )

            if input_data.empty:
                raise ValueError(
                    "Prediction input is empty."
                )

            logger.info(
                f"Prediction input shape: "
                f"{input_data.shape}"
            )

            # --------------------------------------------------
            # Load artifacts FIRST
            # --------------------------------------------------

            (
                preprocessor,
                feature_config,
                model
            ) = self._load_artifacts()

            # --------------------------------------------------
            # Feature engineering
            # --------------------------------------------------

            transformed_input = (
                self._feature_engineering(
                    input_data,
                    feature_config
                )
            )

            logger.info(
                f"Transformed input shape: "
                f"{transformed_input.shape}"
            )

            # --------------------------------------------------
            # Apply fitted preprocessor
            # --------------------------------------------------

            
            processed_input = (
                preprocessor.transform(
                    transformed_input
                )
            )

            # Preserve feature names expected by the trained model
            if hasattr(
                preprocessor,
                "get_feature_names_out"
            ):

                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )

                processed_input = pd.DataFrame(
                    processed_input,
                    columns=feature_names,
                    index=transformed_input.index
                )


            logger.info(
                f"Processed input shape: "
                f"{processed_input.shape}"
            )

            # --------------------------------------------------
            # Predict
            # --------------------------------------------------

            predictions = model.predict(
                processed_input
            )

            predictions = np.asarray(
                predictions
            )

            logger.info(
                "Demand prediction completed successfully."
            )

            return predictions

        except Exception as error:

            logger.error(
                "Error occurred during demand prediction."
            )

            raise CustomException(
                error,
                sys
            )


if __name__ == "__main__":

    pipeline = PredictPipeline()

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

    prediction = pipeline.predict(
        sample_input
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 8 — PREDICTION PIPELINE TEST"
    )

    print(
        "=" * 60
    )

    print(
        f"\nPredicted Demand: "
        f"{prediction[0]:.2f}"
    )

    print(
        "\nPrediction completed successfully."
    )


import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logger


class DataTransformation:
    """
    Handles feature engineering and preprocessing for the
    retail demand prediction dataset.
    """

    def __init__(self):
        self.input_data_path = os.path.join(
            "artifacts",
            "ingested_data",
            "retail_store_inventory.csv"
        )

        self.processed_directory = os.path.join(
            "artifacts",
            "processed"
        )

        self.preprocessor_path = os.path.join(
            self.processed_directory,
            "preprocessor.pkl"
        )

    def _feature_engineering(self, dataframe):
        """
        Apply the feature engineering logic established
        during notebook experimentation.
        """

        logger.info("Starting feature engineering.")

        dataframe = dataframe.copy()

        # --------------------------------------------------
        # Date conversion
        # --------------------------------------------------

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce"
        )

        invalid_dates = dataframe["date"].isna().sum()

        if invalid_dates > 0:
            raise ValueError(
                f"Found {invalid_dates} invalid date values."
            )

        # Sort chronologically
        dataframe = dataframe.sort_values(
            "date"
        ).reset_index(drop=True)

        # --------------------------------------------------
        # Target
        # --------------------------------------------------

        target = "demand"

        if target not in dataframe.columns:
            raise ValueError(
                "Target column 'demand' was not found."
            )

        # --------------------------------------------------
        # Remove potential leakage columns
        # --------------------------------------------------

        leakage_columns = [
            "units_sold",
            "units_ordered"
        ]

        dataframe = dataframe.drop(
            columns=leakage_columns,
            errors="ignore"
        )

        logger.info(
            f"Removed potential leakage columns: "
            f"{leakage_columns}"
        )

        # --------------------------------------------------
        # Date Features
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
        # Cyclical Time Features
        # --------------------------------------------------

        dataframe["month_sin"] = np.sin(
            2 * np.pi * dataframe["month"] / 12
        )

        dataframe["month_cos"] = np.cos(
            2 * np.pi * dataframe["month"] / 12
        )

        dataframe["day_of_week_sin"] = np.sin(
            2 * np.pi * dataframe["day_of_week"] / 7
        )

        dataframe["day_of_week_cos"] = np.cos(
            2 * np.pi * dataframe["day_of_week"] / 7
        )

        # --------------------------------------------------
        # Competitive Pricing Features
        # --------------------------------------------------

        if "competitor_pricing" in dataframe.columns:

            dataframe["price_difference"] = (
                dataframe["price"]
                - dataframe["competitor_pricing"]
            )

            dataframe["price_premium_pct"] = (
                (
                    dataframe["price"]
                    - dataframe["competitor_pricing"]
                )
                / dataframe["competitor_pricing"]
            ) * 100

            dataframe["price_ratio"] = (
                dataframe["price"]
                / dataframe["competitor_pricing"]
            )

            # Remove redundant raw competitor pricing
            dataframe = dataframe.drop(
                columns=["competitor_pricing"],
                errors="ignore"
            )

        # --------------------------------------------------
        # Pricing Level Features
        # --------------------------------------------------

        dataframe["price_level"] = pd.qcut(
            dataframe["price"],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop"
        )

        # --------------------------------------------------
        # Discount Level Features
        # --------------------------------------------------

        dataframe["discount_level"] = pd.qcut(
            dataframe["discount"],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop"
        )

        # --------------------------------------------------
        # Promotion + Discount
        # --------------------------------------------------

        dataframe["promotion_discount"] = (
            dataframe["promotion"]
            * dataframe["discount"]
        )

        # --------------------------------------------------
        # Price + Promotion
        # --------------------------------------------------

        dataframe["price_promotion"] = (
            dataframe["price"]
            * dataframe["promotion"]
        )

        # --------------------------------------------------
        # Price + Discount
        # --------------------------------------------------

        dataframe["price_discount"] = (
            dataframe["price"]
            * dataframe["discount"]
        )

        # --------------------------------------------------
        # Category + Seasonality
        # --------------------------------------------------

        dataframe["category_seasonality"] = (
            dataframe["category"].astype(str)
            + "_"
            + dataframe["seasonality"].astype(str)
        )

        # --------------------------------------------------
        # Category + Price Level
        # --------------------------------------------------

        dataframe["category_price_level"] = (
            dataframe["category"].astype(str)
            + "_"
            + dataframe["price_level"].astype(str)
        )

        # --------------------------------------------------
        # Inventory Transformation
        # --------------------------------------------------

        dataframe["inventory_log"] = np.log1p(
            dataframe["inventory_level"]
        )

        logger.info(
            f"Feature engineering completed. "
            f"Shape: {dataframe.shape}"
        )

        return dataframe

    def _build_preprocessor(self, X_train):
        """
        Build the numerical and categorical preprocessing
        pipelines.
        """

        numerical_features = X_train.select_dtypes(
            include=[
                "int64",
                "int32",
                "float64",
                "float32"
            ]
        ).columns.tolist()

        categorical_features = X_train.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        ).columns.tolist()

        logger.info(
            f"Numerical features: {numerical_features}"
        )

        logger.info(
            f"Categorical features: {categorical_features}"
        )

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numerical_pipeline,
                    numerical_features
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_features
                )
            ],
            remainder="drop"
        )

        return preprocessor

    def initiate_data_transformation(self):
        """
        Perform feature engineering, chronological train/test
        split, preprocessing and artifact creation.
        """

        logger.info(
            "Starting data transformation."
        )

        try:

            # --------------------------------------------------
            # Load Phase 4 ingestion artifact
            # --------------------------------------------------

            if not os.path.exists(
                self.input_data_path
            ):
                raise FileNotFoundError(
                    f"Ingestion file not found: "
                    f"{self.input_data_path}"
                )

            dataframe = pd.read_csv(
                self.input_data_path
            )

            logger.info(
                f"Loaded ingestion data. "
                f"Shape: {dataframe.shape}"
            )

            # --------------------------------------------------
            # Feature Engineering
            # --------------------------------------------------

            dataframe = self._feature_engineering(
                dataframe
            )

            # --------------------------------------------------
            # Chronological Train/Test Split
            # --------------------------------------------------

            dataframe = dataframe.sort_values(
                "date"
            ).reset_index(drop=True)

            split_index = int(
                len(dataframe) * 0.80
            )

            train_df = dataframe.iloc[
                :split_index
            ].copy()

            test_df = dataframe.iloc[
                split_index:
            ].copy()

            logger.info(
                f"Training shape: {train_df.shape}"
            )

            logger.info(
                f"Testing shape: {test_df.shape}"
            )

            logger.info(
                f"Training period: "
                f"{train_df['date'].min()} "
                f"to "
                f"{train_df['date'].max()}"
            )

            logger.info(
                f"Testing period: "
                f"{test_df['date'].min()} "
                f"to "
                f"{test_df['date'].max()}"
            )

            # --------------------------------------------------
            # Separate X and y
            # --------------------------------------------------

            target = "demand"

            X_train = train_df.drop(
                columns=[target]
            )

            y_train = train_df[target]

            X_test = test_df.drop(
                columns=[target]
            )

            y_test = test_df[target]

            # --------------------------------------------------
            # Remove raw date
            # --------------------------------------------------

            X_train = X_train.drop(
                columns=["date"]
            )

            X_test = X_test.drop(
                columns=["date"]
            )

            # --------------------------------------------------
            # Build Preprocessor
            # --------------------------------------------------

            preprocessor = self._build_preprocessor(
                X_train
            )

            # --------------------------------------------------
            # Fit ONLY on training data
            # --------------------------------------------------

            X_train_processed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_test_processed = (
                preprocessor.transform(
                    X_test
                )
            )

            # --------------------------------------------------
            # Feature Names
            # --------------------------------------------------

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            logger.info(
                f"Number of final features: "
                f"{len(feature_names)}"
            )

            # --------------------------------------------------
            # Convert to DataFrames
            # --------------------------------------------------

            X_train_final = pd.DataFrame(
                X_train_processed,
                columns=feature_names
            )

            X_test_final = pd.DataFrame(
                X_test_processed,
                columns=feature_names
            )

            # --------------------------------------------------
            # Create processed directory
            # --------------------------------------------------

            os.makedirs(
                self.processed_directory,
                exist_ok=True
            )

            # --------------------------------------------------
            # Save processed datasets
            # --------------------------------------------------

            X_train_path = os.path.join(
                self.processed_directory,
                "X_train.csv"
            )

            X_test_path = os.path.join(
                self.processed_directory,
                "X_test.csv"
            )

            y_train_path = os.path.join(
                self.processed_directory,
                "y_train.csv"
            )

            y_test_path = os.path.join(
                self.processed_directory,
                "y_test.csv"
            )

            X_train_final.to_csv(
                X_train_path,
                index=False
            )

            X_test_final.to_csv(
                X_test_path,
                index=False
            )

            y_train.to_csv(
                y_train_path,
                index=False
            )

            y_test.to_csv(
                y_test_path,
                index=False
            )

            # --------------------------------------------------
            # Save preprocessor
            # --------------------------------------------------

            joblib.dump(
                preprocessor,
                self.preprocessor_path
            )

            logger.info(
                "Preprocessor saved successfully."
            )

            logger.info(
                "Data transformation completed successfully."
            )

            return (
                X_train_path,
                X_test_path,
                y_train_path,
                y_test_path,
                self.preprocessor_path
            )

        except Exception as error:

            logger.error(
                "Error occurred during data transformation."
            )

            raise CustomException(
                error,
                sys
            )


if __name__ == "__main__":

    transformation = DataTransformation()

    result = (
        transformation
        .initiate_data_transformation()
    )

    print("\nData transformation completed.")

    print("\nGenerated files:")

    for path in result:
        print(path)


import os
import sys
import pandas as pd

from src.database.mysql_connection import create_connection
from src.exception import CustomException
from src.logger import logger


class DataIngestion:
    """
    Handles extraction of retail inventory data from MySQL
    and creation of the ingestion artifact.
    """

    def __init__(self):
        self.ingestion_directory = os.path.join(
            "artifacts",
            "ingested_data"
        )

        self.raw_data_path = os.path.join(
            self.ingestion_directory,
            "retail_store_inventory.csv"
        )

    def initiate_data_ingestion(self):
        """
        Read data from MySQL, validate the extracted data,
        save it as an ingestion artifact, and return the path.
        """

        logger.info("Starting data ingestion.")

        connection = None

        try:
            connection = create_connection()

            if connection is None:
                raise Exception(
                    "Unable to establish MySQL database connection."
                )

            logger.info(
                "Successfully connected to MySQL database."
            )

            query = """
                SELECT
                    date,
                    store_id,
                    product_id,
                    category,
                    region,
                    inventory_level,
                    units_sold,
                    units_ordered,
                    price,
                    discount,
                    weather_condition,
                    promotion,
                    competitor_pricing,
                    seasonality,
                    epidemic,
                    demand
                FROM retail_store_inventory
            """

            logger.info(
                "Reading data from retail_store_inventory table."
            )

            dataframe = pd.read_sql(
                query,
                connection
            )

            logger.info(
                f"Data successfully extracted. "
                f"Shape: {dataframe.shape}"
            )

            if dataframe.empty:
                raise Exception(
                    "The retail_store_inventory table is empty."
                )

            expected_columns = [
                "date",
                "store_id",
                "product_id",
                "category",
                "region",
                "inventory_level",
                "units_sold",
                "units_ordered",
                "price",
                "discount",
                "weather_condition",
                "promotion",
                "competitor_pricing",
                "seasonality",
                "epidemic",
                "demand"
            ]

            missing_columns = [
                column
                for column in expected_columns
                if column not in dataframe.columns
            ]

            if missing_columns:
                raise Exception(
                    f"Missing expected columns: {missing_columns}"
                )

            dataframe["date"] = pd.to_datetime(
                dataframe["date"]
            )

            dataframe = dataframe.sort_values(
                by="date"
            ).reset_index(drop=True)

            os.makedirs(
                self.ingestion_directory,
                exist_ok=True
            )

            dataframe.to_csv(
                self.raw_data_path,
                index=False
            )

            logger.info(
                f"Ingestion artifact saved at: "
                f"{self.raw_data_path}"
            )

            logger.info(
                "Data ingestion completed successfully."
            )

            return self.raw_data_path

        except Exception as error:
            logger.error(
                "Error occurred during data ingestion."
            )

            raise CustomException(
                error,
                sys
            )

        finally:
            if connection is not None:
                connection.close()

                logger.info(
                    "MySQL database connection closed."
                )


if __name__ == "__main__":
    ingestion = DataIngestion()
    output_path = ingestion.initiate_data_ingestion()
    print(
        f"\nData ingestion completed successfully."
    )
    print(
        f"Ingested dataset saved at: {output_path}"
    )


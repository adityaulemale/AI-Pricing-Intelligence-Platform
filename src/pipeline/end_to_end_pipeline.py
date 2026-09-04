import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.pipeline.predict_pipeline import PredictPipeline
from src.optimization.pricing_optimizer import PricingOptimizer
from src.explainability.shap_explainer import SHAPExplainer


class EndToEndPipeline:
    """
    End-to-end integration pipeline.

    Connects:

    1. Demand Prediction
    2. Pricing Optimization
    3. SHAP Explainability

    into a single workflow.
    """

    def __init__(self):

        logger.info(
            "Initializing end-to-end integration pipeline."
        )

        self.predict_pipeline = PredictPipeline()

        self.shap_explainer = SHAPExplainer()

    def run(self, input_data):
        """
        Execute the complete AI pricing workflow.

        Parameters
        ----------
        input_data : dict, pandas.Series, or pandas.DataFrame
            Product/store input data.

        Returns
        -------
        dict
            Complete integrated analysis result.
        """

        logger.info(
            "Starting end-to-end pricing analysis."
        )

        try:

            # ==================================================
            # Validate input
            # ==================================================

            if isinstance(input_data, dict):

                input_dataframe = pd.DataFrame(
                    [input_data]
                )

            elif isinstance(input_data, pd.Series):

                input_dataframe = (
                    input_data.to_frame().T
                )

            elif isinstance(input_data, pd.DataFrame):

                input_dataframe = input_data.copy()

            else:

                raise TypeError(
                    "input_data must be a dictionary, "
                    "Series, or DataFrame."
                )

            if input_dataframe.empty:

                raise ValueError(
                    "End-to-end input is empty."
                )

            if len(input_dataframe) != 1:

                raise ValueError(
                    "End-to-end integration currently "
                    "supports one product/store record "
                    "at a time."
                )

            # ==================================================
            # STEP 1 — Demand Prediction
            # ==================================================

            logger.info(
                "Step 1/3 — Running demand prediction."
            )

            prediction = (
                self.predict_pipeline.predict(
                    input_dataframe
                )
            )

            predicted_demand = max(
                float(prediction[0]),
                0.0
            )

            logger.info(
                f"Predicted demand: "
                f"{predicted_demand:.2f}"
            )

            # ==================================================
            # STEP 2 — Pricing Optimization
            # ==================================================

            logger.info(
                "Step 2/3 — Running pricing optimization."
            )

            current_price = float(
                input_dataframe.iloc[0]["price"]
            )

            optimizer = PricingOptimizer(
                min_price=current_price * 0.80,
                max_price=current_price * 1.20,
                price_step=1.0,
                max_price_change_pct=20.0
            )

            optimization_result = (
                optimizer.optimize(
                    input_dataframe
                )
            )

            logger.info(
                "Pricing optimization completed."
            )

            # ==================================================
            # STEP 3 — SHAP Explainability
            # ==================================================

            logger.info(
                "Step 3/3 — Generating SHAP explanation."
            )

            shap_result = (
                self.shap_explainer.explain_prediction(
                    input_dataframe
                )
            )

            logger.info(
                "SHAP explanation completed."
            )

            # ==================================================
            # Consistency Validation
            # ==================================================

            shap_prediction = float(
                shap_result["predictions"][0]
            )

            prediction_difference = abs(
                predicted_demand
                - shap_prediction
            )

            if prediction_difference > 1e-6:

                raise ValueError(
                    "Prediction mismatch detected between "
                    "PredictPipeline and SHAP explanation. "
                    f"Difference: "
                    f"{prediction_difference:.10f}"
                )

            # ==================================================
            # Build Final Result
            # ==================================================

            result = {

                "input_data":
                    input_dataframe.to_dict(
                        orient="records"
                    )[0],

                "predicted_demand":
                    round(
                        predicted_demand,
                        2
                    ),

                "pricing_optimization":
                    optimization_result,

                "shap_explanation":
                    shap_result,

                "prediction_consistency":
                    {
                        "prediction_pipeline":
                            round(
                                predicted_demand,
                                6
                            ),

                        "shap_prediction":
                            round(
                                shap_prediction,
                                6
                            ),

                        "absolute_difference":
                            round(
                                prediction_difference,
                                10
                            ),

                        "status":
                            "PASS"
                    },

                "status":
                    "SUCCESS"
            }

            logger.info(
                "End-to-end pricing analysis "
                "completed successfully."
            )

            return result

        except Exception as error:

            logger.error(
                "Error occurred during end-to-end "
                "pricing analysis."
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

        sample_input = {

            "date":
                "2023-09-01",

            "store_id":
                "S001",

            "product_id":
                "P001",

            "category":
                "Groceries",

            "region":
                "North",

            "inventory_level":
                500,

            "price":
                50.0,

            "discount":
                10.0,

            "weather_condition":
                "Sunny",

            "promotion":
                1,

            "competitor_pricing":
                48.0,

            "seasonality":
                "Summer",

            "epidemic":
                0
        }

        pipeline = EndToEndPipeline()

        result = pipeline.run(
            sample_input
        )

        # =====================================================
        # Display Results
        # =====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "PHASE 12 — END-TO-END INTEGRATION"
        )

        print(
            "=" * 70
        )

        print(
            "\nSTATUS:",
            result["status"]
        )

        print(
            "\n--- Demand Prediction ---"
        )

        print(
            "Predicted Demand:",
            f"{result['predicted_demand']:.2f}"
        )

        optimization = (
            result[
                "pricing_optimization"
            ]
        )

        print(
            "\n--- Pricing Optimization ---"
        )

        print(
            "Current Price:",
            f"₹{optimization['current_price']:.2f}"
        )

        print(
            "Recommended Price:",
            f"₹{optimization['recommended_price']:.2f}"
        )

        print(
            "Recommendation Status:",
            optimization[
                "recommendation_status"
            ]
        )

        print(
            "Current Revenue:",
            f"₹{optimization['current_revenue']:,.2f}"
        )

        print(
            "Optimized Revenue:",
            f"₹{optimization['optimized_revenue']:,.2f}"
        )

        print(
            "Revenue Improvement:",
            f"{optimization['revenue_improvement_pct']:.2f}%"
        )

        shap_result = (
            result[
                "shap_explanation"
            ]
        )

        print(
            "\n--- SHAP Explainability ---"
        )

        print(
            "SHAP Base Value:",
            f"{shap_result['base_value']:.2f}"
        )

        print(
            "SHAP Prediction:",
            f"{float(shap_result['predictions'][0]):.2f}"
        )

        consistency = (
            result[
                "prediction_consistency"
            ]
        )

        print(
            "\n--- Prediction Consistency ---"
        )

        print(
            "Prediction Pipeline:",
            f"{consistency['prediction_pipeline']:.6f}"
        )

        print(
            "SHAP Prediction:",
            f"{consistency['shap_prediction']:.6f}"
        )

        print(
            "Absolute Difference:",
            f"{consistency['absolute_difference']:.10f}"
        )

        print(
            "Consistency Status:",
            consistency["status"]
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "PHASE 12 — INTEGRATION TEST COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 70
        )

    except Exception as error:

        print(
            "\nPhase 12 integration test failed."
        )

        print(
            f"Error: {error}"
        )

        raise


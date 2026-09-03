import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.pipeline.predict_pipeline import PredictPipeline


class PricingOptimizer:
    """
    Pricing optimization engine.

    Uses the trained demand prediction pipeline to evaluate
    multiple candidate prices and selects the price that
    maximizes expected revenue.

    Revenue = Price × Predicted Demand
    """

    def __init__(
        self,
        min_price=None,
        max_price=None,
        price_step=1.0,
        max_price_change_pct=20.0
    ):
        """
        Initialize the pricing optimizer.

        Parameters
        ----------
        min_price : float, optional
            Minimum allowed price.

        max_price : float, optional
            Maximum allowed price.

        price_step : float, default=1.0
            Difference between consecutive candidate prices.

        max_price_change_pct : float, default=20.0
            Maximum percentage by which the recommended price
            can differ from the current price.
        """

        self.min_price = min_price
        self.max_price = max_price
        self.price_step = price_step

        self.max_price_change_pct = (
            max_price_change_pct
        )

        self.predict_pipeline = PredictPipeline()

    def _generate_candidate_prices(
        self,
        current_price
    ):
        """
        Generate candidate prices while respecting
        business pricing constraints.
        """

        # --------------------------------------------------
        # Validate current price
        # --------------------------------------------------

        if current_price <= 0:
            raise ValueError(
                "Current price must be greater than zero."
            )

        # --------------------------------------------------
        # Validate price step
        # --------------------------------------------------

        if self.price_step <= 0:
            raise ValueError(
                "Price step must be greater than zero."
            )

        # --------------------------------------------------
        # Validate maximum price change
        # --------------------------------------------------

        if self.max_price_change_pct <= 0:
            raise ValueError(
                "Maximum price change percentage "
                "must be greater than zero."
            )

        # --------------------------------------------------
        # Calculate allowed movement
        # --------------------------------------------------

        change_factor = (
            self.max_price_change_pct / 100
        )

        allowed_min_price = (
            current_price
            * (1 - change_factor)
        )

        allowed_max_price = (
            current_price
            * (1 + change_factor)
        )

        # --------------------------------------------------
        # Apply explicit minimum price
        # --------------------------------------------------

        if self.min_price is not None:

            min_price = max(
                float(self.min_price),
                allowed_min_price
            )

        else:

            min_price = allowed_min_price

        # --------------------------------------------------
        # Apply explicit maximum price
        # --------------------------------------------------

        if self.max_price is not None:

            max_price = min(
                float(self.max_price),
                allowed_max_price
            )

        else:

            max_price = allowed_max_price

        # --------------------------------------------------
        # Validate minimum price
        # --------------------------------------------------

        if min_price <= 0:
            raise ValueError(
                "Minimum price must be greater than zero."
            )

        # --------------------------------------------------
        # Validate price range
        # --------------------------------------------------

        if max_price <= min_price:
            raise ValueError(
                "Maximum price must be greater than "
                "minimum price."
            )

        # --------------------------------------------------
        # Generate candidate prices
        # --------------------------------------------------

        candidate_prices = np.arange(
            min_price,
            max_price + self.price_step,
            self.price_step
        )

        candidate_prices = np.round(
            candidate_prices,
            2
        )

        # --------------------------------------------------
        # Always include current price
        # --------------------------------------------------

        candidate_prices = np.append(
            candidate_prices,
            current_price
        )

        # --------------------------------------------------
        # Remove duplicates
        # --------------------------------------------------

        candidate_prices = np.unique(
            candidate_prices
        )

        # --------------------------------------------------
        # Ensure all prices remain within constraints
        # --------------------------------------------------

        candidate_prices = candidate_prices[
            (candidate_prices >= min_price)
            &
            (candidate_prices <= max_price)
        ]

        if len(candidate_prices) == 0:
            raise ValueError(
                "No valid candidate prices were generated."
            )

        logger.info(
            f"Generated {len(candidate_prices)} "
            f"candidate prices between "
            f"{min_price:.2f} and "
            f"{max_price:.2f}."
        )

        return candidate_prices

    @staticmethod
    def _calculate_revenue(
        price,
        predicted_demand
    ):
        """
        Calculate expected revenue.

        Revenue = Price × Predicted Demand
        """

        return (
            price
            * predicted_demand
        )

    @staticmethod
    def _get_recommendation_status(
        recommended_price,
        current_price,
        minimum_candidate_price,
        maximum_candidate_price
    ):
        """
        Determine whether the recommendation is
        optimized internally, unchanged, or limited
        by the pricing boundary.
        """

        if np.isclose(
            recommended_price,
            current_price
        ):
            return "NO_CHANGE"

        if (
            np.isclose(
                recommended_price,
                minimum_candidate_price
            )
            or
            np.isclose(
                recommended_price,
                maximum_candidate_price
            )
        ):
            return "BOUNDARY_LIMITED"

        return "OPTIMIZED"

    def optimize(
        self,
        input_data
    ):
        """
        Find the price that maximizes expected revenue.

        Parameters
        ----------
        input_data:
            Dictionary, Series, or DataFrame containing
            the features required by PredictPipeline.

        Returns
        -------
        dict
            Pricing optimization results.
        """

        logger.info(
            "Starting pricing optimization."
        )

        try:

            # ==================================================
            # Convert input to DataFrame
            # ==================================================

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

            # ==================================================
            # Validate input
            # ==================================================

            if input_dataframe.empty:
                raise ValueError(
                    "Optimization input is empty."
                )

            if len(input_dataframe) != 1:
                raise ValueError(
                    "Pricing optimization currently "
                    "supports one product/store record "
                    "at a time."
                )

            # ==================================================
            # Validate price
            # ==================================================

            if "price" not in input_dataframe.columns:
                raise ValueError(
                    "Input must contain 'price'."
                )

            current_price = float(
                input_dataframe.iloc[0]["price"]
            )

            # ==================================================
            # Generate candidate prices
            # ==================================================

            candidate_prices = (
                self._generate_candidate_prices(
                    current_price
                )
            )

            results = []

            # ==================================================
            # Evaluate candidate prices
            # ==================================================

            for candidate_price in candidate_prices:

                candidate_input = (
                    input_dataframe.copy()
                )

                candidate_input.loc[
                    candidate_input.index[0],
                    "price"
                ] = candidate_price

                # ----------------------------------------------
                # Predict demand
                # ----------------------------------------------

                predicted_demand = (
                    self.predict_pipeline.predict(
                        candidate_input
                    )[0]
                )

                predicted_demand = float(
                    predicted_demand
                )

                # ----------------------------------------------
                # Prevent negative demand
                # ----------------------------------------------

                predicted_demand = max(
                    predicted_demand,
                    0.0
                )

                # ----------------------------------------------
                # Calculate expected revenue
                # ----------------------------------------------

                expected_revenue = (
                    self._calculate_revenue(
                        candidate_price,
                        predicted_demand
                    )
                )

                results.append(
                    {
                        "price": float(
                            candidate_price
                        ),
                        "predicted_demand":
                            predicted_demand,
                        "expected_revenue":
                            float(
                                expected_revenue
                            )
                    }
                )

            # ==================================================
            # Convert results to DataFrame
            # ==================================================

            results_dataframe = pd.DataFrame(
                results
            )

            if results_dataframe.empty:
                raise ValueError(
                    "Pricing optimization produced "
                    "no candidate results."
                )

            # ==================================================
            # Find optimal price
            # ==================================================

            best_row = (
                results_dataframe
                .loc[
                    results_dataframe[
                        "expected_revenue"
                    ].idxmax()
                ]
            )

            recommended_price = float(
                best_row["price"]
            )

            optimized_demand = float(
                best_row["predicted_demand"]
            )

            optimized_revenue = float(
                best_row["expected_revenue"]
            )

            # ==================================================
            # Candidate price boundaries
            # ==================================================

            minimum_candidate_price = float(
                results_dataframe["price"].min()
            )

            maximum_candidate_price = float(
                results_dataframe["price"].max()
            )

            # ==================================================
            # Recommendation status
            # ==================================================

            recommendation_status = (
                self._get_recommendation_status(
                    recommended_price,
                    current_price,
                    minimum_candidate_price,
                    maximum_candidate_price
                )
            )

            # ==================================================
            # Current price metrics
            # ==================================================

            current_rows = (
                results_dataframe[
                    np.isclose(
                        results_dataframe["price"],
                        current_price
                    )
                ]
            )

            if not current_rows.empty:

                current_demand = float(
                    current_rows.iloc[0][
                        "predicted_demand"
                    ]
                )

                current_revenue = float(
                    current_rows.iloc[0][
                        "expected_revenue"
                    ]
                )

            else:

                current_demand = float(
                    self.predict_pipeline.predict(
                        input_dataframe
                    )[0]
                )

                current_demand = max(
                    current_demand,
                    0.0
                )

                current_revenue = (
                    current_price
                    * current_demand
                )

            # ==================================================
            # Revenue improvement
            # ==================================================

            if current_revenue > 0:

                revenue_improvement_pct = (
                    (
                        optimized_revenue
                        - current_revenue
                    )
                    / current_revenue
                ) * 100

            else:

                revenue_improvement_pct = 0.0

            # ==================================================
            # Price change
            # ==================================================

            price_change_pct = (
                (
                    recommended_price
                    - current_price
                )
                / current_price
            ) * 100

            # ==================================================
            # Final optimization result
            # ==================================================

            optimization_result = {

                "current_price":
                    round(
                        current_price,
                        2
                    ),

                "recommended_price":
                    round(
                        recommended_price,
                        2
                    ),

                "recommendation_status":
                    recommendation_status,

                "current_demand":
                    round(
                        current_demand,
                        2
                    ),

                "optimized_demand":
                    round(
                        optimized_demand,
                        2
                    ),

                "current_revenue":
                    round(
                        current_revenue,
                        2
                    ),

                "optimized_revenue":
                    round(
                        optimized_revenue,
                        2
                    ),

                "revenue_improvement_pct":
                    round(
                        revenue_improvement_pct,
                        2
                    ),

                "price_change_pct":
                    round(
                        price_change_pct,
                        2
                    ),

                "minimum_candidate_price":
                    round(
                        minimum_candidate_price,
                        2
                    ),

                "maximum_candidate_price":
                    round(
                        maximum_candidate_price,
                        2
                    ),

                "candidate_prices":
                    results_dataframe
                    .round(2)
                    .to_dict(
                        orient="records"
                    )
            }

            logger.info(
                "Pricing optimization completed successfully."
            )

            return optimization_result

        except Exception as error:

            logger.error(
                "Error occurred during pricing optimization."
            )

            raise CustomException(
                error,
                sys
            )


if __name__ == "__main__":

    # ==========================================================
    # Initialize optimizer
    # ==========================================================

    optimizer = PricingOptimizer(
        min_price=40,
        max_price=60,
        price_step=1,
        max_price_change_pct=20.0
    )

    # ==========================================================
    # Sample product/store input
    # ==========================================================

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

    # ==========================================================
    # Run optimization
    # ==========================================================

    result = optimizer.optimize(
        sample_input
    )

    # ==========================================================
    # Display summary
    # ==========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 9 — PRICING OPTIMIZATION"
    )

    print(
        "=" * 60
    )

    print(
        f"\nCurrent Price: "
        f"{result['current_price']:.2f}"
    )

    print(
        f"Recommended Price: "
        f"{result['recommended_price']:.2f}"
    )

    print(
        f"Recommendation Status: "
        f"{result['recommendation_status']}"
    )

    print(
        f"\nCurrent Demand: "
        f"{result['current_demand']:.2f}"
    )

    print(
        f"Optimized Demand: "
        f"{result['optimized_demand']:.2f}"
    )

    print(
        f"\nCurrent Revenue: "
        f"{result['current_revenue']:.2f}"
    )

    print(
        f"Optimized Revenue: "
        f"{result['optimized_revenue']:.2f}"
    )

    print(
        f"\nRevenue Improvement: "
        f"{result['revenue_improvement_pct']:.2f}%"
    )

    print(
        f"Price Change: "
        f"{result['price_change_pct']:.2f}%"
    )

    print(
        f"\nAllowed Candidate Range: "
        f"₹{result['minimum_candidate_price']:.2f}"
        f" - "
        f"₹{result['maximum_candidate_price']:.2f}"
    )

    # ==========================================================
    # Price sensitivity analysis
    # ==========================================================

    print(
        "\nPrice Sensitivity Analysis:"
    )

    print(
        "\nPrice | Predicted Demand | Expected Revenue"
    )

    print(
        "-" * 50
    )

    for row in result["candidate_prices"]:

        print(
            f"₹{row['price']:>6.2f} | "
            f"{row['predicted_demand']:>16.2f} | "
            f"₹{row['expected_revenue']:>16.2f}"
        )

    print(
        "\n" + "=" * 60
    )


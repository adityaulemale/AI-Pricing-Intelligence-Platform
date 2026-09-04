import os
import sys

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.pipeline.predict_pipeline import (
    PredictPipeline
)

from src.optimization.pricing_optimizer import (
    PricingOptimizer
)

from src.explainability.shap_explainer import (
    SHAPExplainer
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Pricing Intelligence Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
    }

    .recommendation-box {
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #dddddd;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .success-text {
        font-size: 24px;
        font-weight: 700;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '💰 AI Pricing Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Demand Prediction • Dynamic Pricing • Revenue Optimization '
    '• Explainable AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Input Parameters"
)

st.sidebar.markdown(
    "Enter the product/store information below."
)


# ============================================================
# INPUT SECTION
# ============================================================

date = st.sidebar.date_input(
    "Date",
    value=pd.Timestamp(
        "2023-09-01"
    ).date()
)

store_id = st.sidebar.text_input(
    "Store ID",
    value="S001"
)

product_id = st.sidebar.text_input(
    "Product ID",
    value="P001"
)

category = st.sidebar.selectbox(
    "Category",
    [
        "Groceries",
        "Furniture",
        "Electronics"
    ]
)

region = st.sidebar.selectbox(
    "Region",
    [
        "North",
        "South",
        "East",
        "West"
    ]
)

inventory_level = st.sidebar.number_input(
    "Inventory Level",
    min_value=0.0,
    value=500.0,
    step=1.0
)

price = st.sidebar.number_input(
    "Current Price",
    min_value=0.01,
    value=50.0,
    step=1.0
)

discount = st.sidebar.number_input(
    "Discount (%)",
    min_value=0.0,
    value=10.0,
    step=1.0
)

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    [
        "Sunny",
        "Rainy",
        "Cloudy",
        "Snowy"
    ]
)

promotion = st.sidebar.selectbox(
    "Promotion",
    [
        0,
        1
    ],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)

competitor_pricing = st.sidebar.number_input(
    "Competitor Price",
    min_value=0.01,
    value=48.0,
    step=1.0
)

seasonality = st.sidebar.selectbox(
    "Seasonality",
    [
        "Summer",
        "Winter",
        "Spring",
        "Autumn"
    ]
)

epidemic = st.sidebar.selectbox(
    "Epidemic",
    [
        0,
        1
    ],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)


# ============================================================
# BUILD INPUT DATA
# ============================================================

input_data = {

    "date":
        str(date),

    "store_id":
        store_id,

    "product_id":
        product_id,

    "category":
        category,

    "region":
        region,

    "inventory_level":
        inventory_level,

    "price":
        price,

    "discount":
        discount,

    "weather_condition":
        weather_condition,

    "promotion":
        promotion,

    "competitor_pricing":
        competitor_pricing,

    "seasonality":
        seasonality,

    "epidemic":
        epidemic
}


# ============================================================
# ACTION BUTTON
# ============================================================

run_analysis = st.sidebar.button(
    "🚀 Analyze Pricing",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if run_analysis:

    try:

        # ====================================================
        # LOAD COMPONENTS
        # ====================================================

        with st.spinner(
            "Running AI pricing analysis..."
        ):

            prediction_pipeline = (
                PredictPipeline()
            )

            optimizer = (
                PricingOptimizer(
                    min_price=price * 0.80,
                    max_price=price * 1.20,
                    price_step=1.0,
                    max_price_change_pct=20.0
                )
            )

            shap_explainer = (
                SHAPExplainer()
            )


        # ====================================================
        # DEMAND PREDICTION
        # ====================================================

        with st.spinner(
            "Predicting demand..."
        ):

            prediction = (
                prediction_pipeline.predict(
                    input_data
                )
            )

            predicted_demand = float(
                prediction[0]
            )

            predicted_demand = max(
                predicted_demand,
                0.0
            )


        # ====================================================
        # PRICING OPTIMIZATION
        # ====================================================

        with st.spinner(
            "Optimizing price..."
        ):

            optimization_result = (
                optimizer.optimize(
                    input_data
                )
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        with st.spinner(
            "Generating explainability results..."
        ):

            shap_result = (
                shap_explainer.explain_prediction(
                    input_data
                )
            )


        # ====================================================
        # STORE RESULTS IN SESSION STATE
        # ====================================================

        st.session_state[
            "predicted_demand"
        ] = predicted_demand

        st.session_state[
            "optimization_result"
        ] = optimization_result

        st.session_state[
            "shap_result"
        ] = shap_result

        st.session_state[
            "analysis_completed"
        ] = True


    except Exception as error:

        st.error(
            "Unable to complete pricing analysis."
        )

        st.exception(
            error
        )

        st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.get(
    "analysis_completed",
    False
):

    predicted_demand = (
        st.session_state[
            "predicted_demand"
        ]
    )

    optimization_result = (
        st.session_state[
            "optimization_result"
        ]
    )

    shap_result = (
        st.session_state[
            "shap_result"
        ]
    )


    # ========================================================
    # SECTION 1 — DEMAND PREDICTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Demand Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted Demand",
            f"{predicted_demand:.2f}"
        )

    with col2:

        st.metric(
            "Current Price",
            f"₹{price:.2f}"
        )

    with col3:

        current_revenue = (
            price
            * predicted_demand
        )

        st.metric(
            "Expected Current Revenue",
            f"₹{current_revenue:,.2f}"
        )


    # ========================================================
    # SECTION 2 — PRICING RECOMMENDATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '💰 Pricing Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    recommended_price = (
        optimization_result[
            "recommended_price"
        ]
    )

    optimized_demand = (
        optimization_result[
            "optimized_demand"
        ]
    )

    optimized_revenue = (
        optimization_result[
            "optimized_revenue"
        ]
    )

    revenue_improvement = (
        optimization_result[
            "revenue_improvement_pct"
        ]
    )

    price_change = (
        optimization_result[
            "price_change_pct"
        ]
    )

    recommendation_status = (
        optimization_result[
            "recommendation_status"
        ]
    )


    # --------------------------------------------------------
    # Recommendation box
    # --------------------------------------------------------

    st.markdown(
        '<div class="recommendation-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="success-text">'
        f'Recommended Price: ₹{recommended_price:.2f}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Recommendation Status: "
        f"**{recommendation_status}**"
    )

    st.write(
        f"Price Change: "
        f"**{price_change:.2f}%**"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # OPTIMIZATION METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Demand",
            f"{optimization_result['current_demand']:.2f}"
        )

    with col2:

        st.metric(
            "Optimized Demand",
            f"{optimized_demand:.2f}"
        )

    with col3:

        st.metric(
            "Current Revenue",
            f"₹{optimization_result['current_revenue']:,.2f}"
        )

    with col4:

        st.metric(
            "Optimized Revenue",
            f"₹{optimized_revenue:,.2f}",
            delta=f"{revenue_improvement:.2f}%"
        )


    # ========================================================
    # SECTION 3 — PRICE SENSITIVITY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Price Sensitivity Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    candidate_prices = (
        optimization_result[
            "candidate_prices"
        ]
    )

    sensitivity_dataframe = (
        pd.DataFrame(
            candidate_prices
        )
    )

    if not sensitivity_dataframe.empty:

        st.dataframe(
            sensitivity_dataframe,
            use_container_width=True,
            hide_index=True,
            column_config={
                "price":
                    st.column_config.NumberColumn(
                        "Price",
                        format="₹%.2f"
                    ),

                "predicted_demand":
                    st.column_config.NumberColumn(
                        "Predicted Demand",
                        format="%.2f"
                    ),

                "expected_revenue":
                    st.column_config.NumberColumn(
                        "Expected Revenue",
                        format="₹%.2f"
                    )
            }
        )

        # ----------------------------------------------------
        # Revenue chart
        # ----------------------------------------------------

        chart_dataframe = (
            sensitivity_dataframe
            .set_index(
                "price"
            )[
                [
                    "expected_revenue"
                ]
            ]
        )

        st.line_chart(
            chart_dataframe
        )


    # ========================================================
    # SECTION 4 — SHAP EXPLAINABILITY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔍 Explainable AI — SHAP'
        '</div>',
        unsafe_allow_html=True
    )

    shap_predictions = (
        shap_result[
            "predictions"
        ]
    )

    shap_base_value = (
        shap_result[
            "base_value"
        ]
    )

    st.write(
        f"**SHAP Base Value:** "
        f"{shap_base_value:.2f}"
    )

    st.write(
        f"**SHAP Prediction:** "
        f"{float(shap_predictions[0]):.2f}"
    )


    # ========================================================
    # SHAP GLOBAL IMPORTANCE
    # ========================================================

    st.subheader(
        "Global Feature Importance"
    )

    feature_importance = (
        shap_result[
            "feature_importance"
        ]
        .head(10)
        .copy()
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "mean_abs_shap"
        )
    )

    st.bar_chart(
        feature_importance.set_index(
            "feature"
        )[
            [
                "mean_abs_shap"
            ]
        ]
    )


    # ========================================================
    # SHAP LOCAL EXPLANATION
    # ========================================================

    st.subheader(
        "Prediction Drivers"
    )

    local_explanation = (
        shap_result[
            "individual_explanations"
        ][0]
        .head(10)
        .copy()
    )

    st.dataframe(
        local_explanation,
        use_container_width=True,
        hide_index=True,
        column_config={

            "feature":
                st.column_config.TextColumn(
                    "Feature"
                ),

            "feature_value":
                st.column_config.NumberColumn(
                    "Feature Value",
                    format="%.4f"
                ),

            "shap_value":
                st.column_config.NumberColumn(
                    "SHAP Value",
                    format="%.4f"
                ),

            "absolute_shap_value":
                st.column_config.NumberColumn(
                    "Absolute SHAP",
                    format="%.4f"
                )
        }
    )


    # ========================================================
    # SHAP LOCAL CHART
    # ========================================================

    local_chart = (
        local_explanation[
            [
                "feature",
                "shap_value"
            ]
        ]
        .set_index(
            "feature"
        )
        .sort_values(
            "shap_value"
        )
    )

    st.bar_chart(
        local_chart
    )


    # ========================================================
    # GENERATED SHAP PLOTS
    # ========================================================

    st.subheader(
        "SHAP Visualizations"
    )

    global_plot_path = os.path.join(
        "artifacts",
        "explainability",
        "shap_feature_importance.png"
    )

    local_plot_path = os.path.join(
        "artifacts",
        "explainability",
        "shap_prediction_explanation.png"
    )

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:

        if os.path.exists(
            global_plot_path
        ):

            st.image(
                global_plot_path,
                caption="Global SHAP Feature Importance",
                use_container_width=True
            )

        else:

            st.info(
                "Global SHAP plot has not been generated."
            )

    with plot_col2:

        if os.path.exists(
            local_plot_path
        ):

            st.image(
                local_plot_path,
                caption="Local SHAP Prediction Explanation",
                use_container_width=True
            )

        else:

            st.info(
                "Local SHAP plot has not been generated."
            )


    # ========================================================
    # SECTION 5 — INPUT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Input Summary'
        '</div>',
        unsafe_allow_html=True
    )

    input_summary = (
        pd.DataFrame(
            {
                "Parameter":
                    list(
                        input_data.keys()
                    ),

                "Value":
                    list(
                        input_data.values()
                    )
            }
        )
    )

    st.dataframe(
        input_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "Enter the product/store parameters in the "
        "sidebar and click **Analyze Pricing** "
        "to start the AI analysis."
    )

    st.markdown(
        """
        ### 🚀 Platform Capabilities

        This application combines:

        - 📈 **Demand Prediction**
        - 💰 **Dynamic Pricing Optimization**
        - 📊 **Revenue Improvement Analysis**
        - 🔍 **SHAP Explainability**
        - 📋 **Price Sensitivity Analysis**

        The system uses the trained demand model and the
        same preprocessing and feature-engineering pipeline
        developed during the previous project phases.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "AI Pricing Intelligence Platform | "
    "Demand Prediction + Revenue Optimization + Explainable AI"
)


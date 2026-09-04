import streamlit as st


def display_prediction_metrics(
    current_demand,
    optimized_demand,
    current_revenue,
    optimized_revenue
):
    """
    Display demand and revenue metrics.
    """

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Demand",
            f"{current_demand:.2f}"
        )

    with col2:
        st.metric(
            "Optimized Demand",
            f"{optimized_demand:.2f}"
        )

    with col3:
        st.metric(
            "Current Revenue",
            f"₹{current_revenue:,.2f}"
        )

    with col4:
        st.metric(
            "Optimized Revenue",
            f"₹{optimized_revenue:,.2f}"
        )


def display_price_recommendation(
    current_price,
    recommended_price,
    revenue_improvement,
    recommendation_status
):
    """
    Display pricing recommendation.
    """

    st.subheader(
        "Pricing Recommendation"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Price",
            f"₹{current_price:.2f}"
        )

    with col2:
        st.metric(
            "Recommended Price",
            f"₹{recommended_price:.2f}"
        )

    with col3:
        st.metric(
            "Revenue Improvement",
            f"{revenue_improvement:.2f}%"
        )

    with col4:
        st.metric(
            "Status",
            recommendation_status
        )


def display_shap_importance(
    feature_importance
):
    """
    Display SHAP feature importance.
    """

    st.subheader(
        "Demand Prediction Drivers"
    )

    st.dataframe(
        feature_importance.head(10),
        use_container_width=True,
        hide_index=True
    )


def display_price_sensitivity(
    sensitivity_dataframe
):
    """
    Display price sensitivity analysis.
    """

    st.subheader(
        "Price Sensitivity Analysis"
    )

    st.dataframe(
        sensitivity_dataframe,
        use_container_width=True,
        hide_index=True
    )


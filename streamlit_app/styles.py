import streamlit as st


def apply_custom_styles():
    """
    Apply custom CSS styling to the Streamlit application.
    """

    st.markdown(
        """
        <style>

        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            text-align: center;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


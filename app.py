import streamlit as st

st.set_page_config(
    page_title="Marketing Campaign Performance Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Marketing Campaign Performance Prediction")

st.markdown("---")

st.markdown(
    """
    ## Welcome

    This dashboard provides an end-to-end analysis of marketing campaign performance using
    Data Analytics and Machine Learning.

    ### Features

    - 📊 Dataset Overview
    - 📈 Exploratory Data Analysis
    - 💰 Revenue Prediction (Regression)
    - 🎯 Profit/Loss Prediction (Classification)
    - 📋 Model Comparison
    - 💡 Business Insights
    - ℹ️ About Project

    Use the navigation menu on the left to explore each section.
    """
)

st.markdown("---")

st.info(
    "👈 Select a page from the sidebar to begin exploring the project."
)
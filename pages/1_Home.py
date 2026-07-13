import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Marketing Campaign Performance Prediction")

st.markdown("---")

st.header("📌 Project Overview")

st.write(
    """
    The **Marketing Campaign Performance Prediction** project analyzes marketing campaign
    data to discover business insights and build Machine Learning models for predicting
    campaign revenue and campaign profitability.

    The project combines **Data Analytics**, **Exploratory Data Analysis (EDA)**,
    **Feature Engineering**, and **Machine Learning** into an interactive Streamlit dashboard.
    """
)

st.markdown("---")

st.header("🎯 Project Objectives")

st.markdown("""
- Analyze marketing campaign performance.
- Identify high-performing brands and campaign types.
- Perform Exploratory Data Analysis (EDA).
- Predict campaign revenue using Regression models.
- Predict campaign profit/loss using Classification models.
- Compare multiple Machine Learning models.
- Support business decision-making using data-driven insights.
""")

st.markdown("---")

st.header("📂 Dataset Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", "Marketing Campaign")

with col2:
    st.metric("Target (Regression)", "Revenue")

with col3:
    st.metric("Target (Classification)", "Profit / Loss")

st.markdown("---")

st.header("🛠️ Technologies Used")

st.markdown("""
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Streamlit
""")

st.markdown("---")

st.header("🤖 Machine Learning Models")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Regression")
    st.markdown("""
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    """)

with col2:
    st.subheader("Classification")
    st.markdown("""
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    """)

st.markdown("---")

st.header("📈 Project Workflow")

st.markdown("""
1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Exploratory Data Analysis
5. Regression Modeling
6. Classification Modeling
7. Model Evaluation
8. Model Comparison
9. Business Insights
10. Interactive Streamlit Dashboard
""")

st.markdown("---")

st.success(
    "👈 Use the sidebar to navigate through the dashboard pages."
)
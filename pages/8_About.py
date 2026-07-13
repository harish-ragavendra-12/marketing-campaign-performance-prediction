import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About the Project")

st.markdown("---")

# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

st.header("📌 Project Overview")

st.write("""
The **Marketing Campaign Performance Prediction** project is an end-to-end
Data Science application that analyzes marketing campaign performance and
predicts campaign revenue and profitability using Machine Learning models.

The project combines Data Analysis, Feature Engineering, Exploratory Data
Analysis (EDA), Machine Learning, and an interactive Streamlit dashboard
to support business decision-making.
""")

st.markdown("---")

# ==========================================================
# BUSINESS OBJECTIVES
# ==========================================================

st.header("🎯 Business Objectives")

st.markdown("""
- Analyze marketing campaign performance.
- Identify high-performing campaigns.
- Discover customer behavior patterns.
- Predict campaign revenue.
- Predict campaign profit or loss.
- Compare Machine Learning models.
- Generate business insights for decision-making.
""")

st.markdown("---")

# ==========================================================
# TECHNOLOGY STACK
# ==========================================================

st.header("🛠 Technology Stack")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Programming")

    st.markdown("""
- Python
- Pandas
- NumPy
""")

    st.subheader("Visualization")

    st.markdown("""
- Matplotlib
- Seaborn
- Streamlit
""")

with col2:

    st.subheader("Machine Learning")

    st.markdown("""
- Scikit-Learn
- Joblib
""")

    st.subheader("Development")

    st.markdown("""
- PyCharm
- Git
- GitHub
""")

st.markdown("---")

# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

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

# ==========================================================
# PROJECT WORKFLOW
# ==========================================================

st.header("📊 Project Workflow")

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

10. Interactive Dashboard
""")

st.markdown("---")

# ==========================================================
# DATASET FEATURES
# ==========================================================

st.header("📂 Dataset Features")

st.markdown("""
- Campaign Type
- Target Audience
- Language
- Customer Segment
- Brand
- Duration
- Impressions
- Clicks
- Leads
- Conversions
- Acquisition Cost
- Engagement Score
- Marketing Channels
- Revenue
- ROI
- Profit / Loss
""")

st.markdown("---")

# ==========================================================
# PROJECT HIGHLIGHTS
# ==========================================================

st.header("⭐ Project Highlights")

st.success("✔ End-to-End Data Science Project")

st.success("✔ Interactive Streamlit Dashboard")

st.success("✔ Regression & Classification Models")

st.success("✔ Automated Model Comparison")

st.success("✔ Business Insight Generation")

st.success("✔ Model Persistence using Joblib")

st.success("✔ Professional Project Structure")

st.markdown("---")

# ==========================================================
# FUTURE IMPROVEMENTS
# ==========================================================

st.header("🚀 Future Improvements")

st.markdown("""
- Hyperparameter tuning using GridSearchCV.
- Cross-validation for improved model evaluation.
- Model deployment on Streamlit Cloud.
- Real-time campaign prediction using APIs.
- Interactive Power BI dashboard.
- Automated report generation.
""")

st.markdown("---")

# ==========================================================
# DEVELOPER
# ==========================================================

st.header("👨‍💻 Developer")

st.info("""
**Developed by:**

Harish Ragavendra

Data Science Portfolio Project

2026
""")
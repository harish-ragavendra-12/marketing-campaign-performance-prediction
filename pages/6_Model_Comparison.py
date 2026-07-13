import streamlit as st
import pandas as pd

from config import REPORTS_DIR

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📋",
    layout="wide"
)


# ==========================================================
# LOAD RESULTS
# ==========================================================

@st.cache_data
def load_regression_results():

    return pd.read_csv(
        REPORTS_DIR / "regression_results.csv"
    )


@st.cache_data
def load_classification_results():

    return pd.read_csv(
        REPORTS_DIR / "classification_results.csv"
    )


regression_df = load_regression_results()

classification_df = load_classification_results()


# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("📋 Model Comparison")

st.markdown("---")


# ==========================================================
# REGRESSION RESULTS
# ==========================================================

st.header("💰 Regression Models")

st.dataframe(
    regression_df,
    use_container_width=True
)

best_regression = regression_df.loc[
    regression_df["R2 Score"].idxmax()
]

st.success(
    f"🏆 Best Regression Model : {best_regression['Model']}"
)

st.metric(
    "Best R² Score",
    f"{best_regression['R2 Score']:.4f}"
)

st.markdown("---")


# ==========================================================
# REGRESSION BAR CHART
# ==========================================================

st.subheader("Regression Model Comparison")

st.bar_chart(
    regression_df.set_index("Model")["R2 Score"]
)

st.markdown("---")


# ==========================================================
# CLASSIFICATION RESULTS
# ==========================================================

st.header("🎯 Classification Models")

st.dataframe(
    classification_df,
    use_container_width=True
)

best_classification = classification_df.loc[
    classification_df["F1 Score"].idxmax()
]

st.success(
    f"🏆 Best Classification Model : {best_classification['Model']}"
)

st.metric(
    "Best F1 Score",
    f"{best_classification['F1 Score']:.4f}"
)

st.markdown("---")


# ==========================================================
# CLASSIFICATION BAR CHART
# ==========================================================

st.subheader("Classification Model Comparison")

st.bar_chart(
    classification_df.set_index("Model")["F1 Score"]
)

st.markdown("---")


# ==========================================================
# FINAL SUMMARY
# ==========================================================

st.header("📌 Summary")

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"""
**Best Regression Model**

{best_regression['Model']}

R² Score : {best_regression['R2 Score']:.4f}
"""
    )

with col2:

    st.info(
        f"""
**Best Classification Model**

{best_classification['Model']}

F1 Score : {best_classification['F1 Score']:.4f}
"""
    )
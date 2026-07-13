import streamlit as st
import pandas as pd

from config import PROCESSED_DATA_DIR

st.set_page_config(
    page_title="Dataset Overview",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():

    file_path = (
        PROCESSED_DATA_DIR /
        "feature_engineered_marketing_campaign.csv"
    )

    df = pd.read_csv(file_path)

    return df


df = load_data()

st.title("📊 Dataset Overview")

st.markdown("---")

# ==========================================================
# Dataset Metrics
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

st.markdown("---")

# ==========================================================
# First Five Records
# ==========================================================

st.subheader("First Five Records")

st.dataframe(
    df.head(),
    use_container_width=True
)

# ==========================================================
# Last Five Records
# ==========================================================

st.subheader("Last Five Records")

st.dataframe(
    df.tail(),
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# Column Names
# ==========================================================

st.subheader("Column Names")

st.write(df.columns.tolist())

st.markdown("---")

# ==========================================================
# Data Types
# ==========================================================

st.subheader("Data Types")

datatype_df = pd.DataFrame(
    {
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    }
)

st.dataframe(
    datatype_df,
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# Missing Values
# ==========================================================

st.subheader("Missing Values")

missing_df = pd.DataFrame(
    {
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    }
)

st.dataframe(
    missing_df,
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# Statistical Summary
# ==========================================================

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(include="all"),
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# Download Dataset
# ==========================================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name="feature_engineered_marketing_campaign.csv",
    mime="text/csv"
)
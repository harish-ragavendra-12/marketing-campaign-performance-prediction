import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_DIR

st.set_page_config(
    page_title="EDA",
    page_icon="📈",
    layout="wide"
)


@st.cache_data
def load_data():

    file_path = (
        PROCESSED_DATA_DIR /
        "feature_engineered_marketing_campaign.csv"
    )

    return pd.read_csv(file_path)


df = load_data()

st.title("📈 Exploratory Data Analysis")

st.markdown("---")

##############################################################
# Histogram
##############################################################

st.subheader("Histogram")

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

hist_column = st.selectbox(
    "Select Numerical Column",
    numeric_columns
)

fig, ax = plt.subplots(figsize=(8,5))

ax.hist(
    df[hist_column],
    bins=30,
    edgecolor="black"
)

ax.set_title(f"{hist_column} Distribution")
ax.set_xlabel(hist_column)
ax.set_ylabel("Frequency")

st.pyplot(fig)

st.markdown("---")

##############################################################
# Box Plot
##############################################################

st.subheader("Box Plot")

box_column = st.selectbox(
    "Select Column for Box Plot",
    numeric_columns,
    key="box"
)

fig, ax = plt.subplots(figsize=(8,5))

ax.boxplot(df[box_column])

ax.set_title(f"{box_column} Box Plot")
ax.set_ylabel(box_column)

st.pyplot(fig)

st.markdown("---")

##############################################################
# Count Plot
##############################################################

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

categorical_columns.remove("Campaign_ID")

st.subheader("Count Plot")

count_column = st.selectbox(
    "Select Categorical Column",
    categorical_columns
)

fig, ax = plt.subplots(figsize=(10,5))

sns.countplot(
    data=df,
    x=count_column,
    ax=ax
)

plt.xticks(rotation=45)

st.pyplot(fig)

st.markdown("---")

##############################################################
# Scatter Plot
##############################################################

st.subheader("Scatter Plot")

col1, col2 = st.columns(2)

with col1:

    x_axis = st.selectbox(
        "X-axis",
        numeric_columns,
        key="scatter_x"
    )

with col2:

    y_axis = st.selectbox(
        "Y-axis",
        numeric_columns,
        index=1,
        key="scatter_y"
    )

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(
    df[x_axis],
    df[y_axis]
)

ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_title(f"{x_axis} vs {y_axis}")

st.pyplot(fig)

st.markdown("---")

##############################################################
# Correlation Heatmap
##############################################################

st.subheader("Correlation Heatmap")

corr = df[numeric_columns].corr()

fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

st.pyplot(fig)

st.markdown("---")

##############################################################
# Brand Performance
##############################################################

st.subheader("Average Revenue by Brand")

brand_df = (
    df.groupby("Brand")["Revenue"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(
    brand_df["Brand"],
    brand_df["Revenue"],
    edgecolor="black"
)

plt.xticks(rotation=15)

st.pyplot(fig)

st.markdown("---")

##############################################################
# Campaign Performance
##############################################################

st.subheader("Average Revenue by Campaign Type")

campaign_df = (
    df.groupby("Campaign_Type")["Revenue"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(
    campaign_df["Campaign_Type"],
    campaign_df["Revenue"],
    edgecolor="black"
)

plt.xticks(rotation=15)

st.pyplot(fig)
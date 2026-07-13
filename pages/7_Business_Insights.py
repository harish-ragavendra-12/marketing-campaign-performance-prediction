import streamlit as st
import pandas as pd

from config import PROCESSED_DATA_DIR

st.set_page_config(
    page_title="Business Insights",
    page_icon="💡",
    layout="wide"
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    file_path = (
        PROCESSED_DATA_DIR /
        "feature_engineered_marketing_campaign.csv"
    )

    return pd.read_csv(file_path)


df = load_data()

st.title("💡 Business Insights")

st.markdown("---")


# ==========================================================
# KPI CALCULATIONS
# ==========================================================

highest_revenue = df["Revenue"].max()

average_revenue = df["Revenue"].mean()

highest_roi = df["ROI"].max()

average_roi = df["ROI"].mean()

profit_campaigns = (
    df["Profit_Loss"] == "Profit"
).sum()

loss_campaigns = (
    df["Profit_Loss"] == "Loss"
).sum()


# ==========================================================
# KPI CARDS
# ==========================================================

st.header("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Highest Revenue",
        f"₹ {highest_revenue:,.2f}"
    )

with col2:

    st.metric(
        "Average Revenue",
        f"₹ {average_revenue:,.2f}"
    )

with col3:

    st.metric(
        "Highest ROI",
        f"{highest_roi:.2f}"
    )

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Average ROI",
        f"{average_roi:.2f}"
    )

with col2:

    st.metric(
        "Profit Campaigns",
        profit_campaigns
    )

with col3:

    st.metric(
        "Loss Campaigns",
        loss_campaigns
    )

st.markdown("---")


# ==========================================================
# BEST BRAND
# ==========================================================

st.header("🏆 Best Performing Brand")

brand_df = (
    df.groupby("Brand")
      .agg(
          Average_Revenue=("Revenue", "mean"),
          Average_ROI=("ROI", "mean")
      )
      .sort_values(
          by="Average_Revenue",
          ascending=False
      )
)

st.dataframe(
    brand_df,
    use_container_width=True
)

st.markdown("---")


# ==========================================================
# BEST CAMPAIGN TYPE
# ==========================================================

st.header("📈 Campaign Type Performance")

campaign_df = (
    df.groupby("Campaign_Type")
      .agg(
          Average_Revenue=("Revenue", "mean"),
          Average_ROI=("ROI", "mean")
      )
      .sort_values(
          by="Average_Revenue",
          ascending=False
      )
)

st.dataframe(
    campaign_df,
    use_container_width=True
)

st.bar_chart(
    campaign_df["Average_Revenue"]
)

st.markdown("---")


# ==========================================================
# CUSTOMER SEGMENT
# ==========================================================

st.header("👥 Customer Segment Performance")

segment_df = (
    df.groupby("Customer_Segment")
      .agg(
          Average_Revenue=("Revenue", "mean")
      )
      .sort_values(
          by="Average_Revenue",
          ascending=False
      )
)

st.dataframe(
    segment_df,
    use_container_width=True
)

st.bar_chart(
    segment_df["Average_Revenue"]
)

st.markdown("---")


# ==========================================================
# BUSINESS RECOMMENDATIONS
# ==========================================================

st.header("📌 Business Recommendations")

st.success("""
✅ Invest more in the highest-performing campaign types.
""")

st.success("""
✅ Prioritize high-performing customer segments.
""")

st.success("""
✅ Allocate marketing budget to brands with consistently higher ROI.
""")

st.success("""
✅ Monitor campaigns with negative ROI and optimize acquisition costs.
""")

st.success("""
✅ Improve conversion rates through better audience targeting.
""")
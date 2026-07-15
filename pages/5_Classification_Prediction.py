import streamlit as st
import pandas as pd
import joblib

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR
)

st.set_page_config(
    page_title="Classification Prediction",
    page_icon="🎯",
    layout="wide"
)


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    file_path = (
        PROCESSED_DATA_DIR /
        "feature_engineered_marketing_campaign.csv"
    )

    df = pd.read_csv(file_path)

    return df


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODELS_DIR /
        "best_classification_model.pkl"
    )

    return model


# ==========================================================
# LOAD PREPROCESSOR
# ==========================================================

@st.cache_resource
def load_preprocessor():

    preprocessor = joblib.load(
        MODELS_DIR /
        "classification_preprocessor.pkl"
    )

    return preprocessor


# ==========================================================
# LOAD EVERYTHING
# ==========================================================

df = load_dataset()

model = load_model()

preprocessor = load_preprocessor()


# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("🎯 Profit / Loss Prediction")

st.markdown(
    """
Predict whether a marketing campaign is expected to result in
**Profit** or **Loss** using the trained Machine Learning model.
"""
)

st.markdown("---")


# ==========================================================
# GET UNIQUE VALUES
# ==========================================================

campaign_types = sorted(
    df["Campaign_Type"].unique()
)

target_audience = sorted(
    df["Target_Audience"].unique()
)

languages = sorted(
    df["Language"].unique()
)

customer_segments = sorted(
    df["Customer_Segment"].unique()
)

brands = sorted(
    df["Brand"].unique()
)

duration_categories = sorted(
    df["Duration_Category"].unique()
)

# ==========================================================
# USER INPUT
# ==========================================================

st.subheader("Enter Campaign Details")

col1, col2 = st.columns(2)

with col1:

    campaign_type = st.selectbox(
        "Campaign Type",
        campaign_types
    )

    target = st.selectbox(
        "Target Audience",
        target_audience
    )

    language = st.selectbox(
        "Language",
        languages
    )

    customer_segment = st.selectbox(
        "Customer Segment",
        customer_segments
    )

    brand = st.selectbox(
        "Brand",
        brands
    )

    duration_category = st.selectbox(
        "Duration Category",
        duration_categories
    )

    duration = st.number_input(
        "Duration (Days)",
        min_value=1,
        value=30
    )

with col2:

    impressions = st.number_input(
        "Impressions",
        min_value=0,
        value=10000
    )

    clicks = st.number_input(
        "Clicks",
        min_value=0,
        value=500
    )

    leads = st.number_input(
        "Leads",
        min_value=0,
        value=100
    )

    conversions = st.number_input(
        "Conversions",
        min_value=0,
        value=20
    )

    acquisition_cost = st.number_input(
        "Acquisition Cost",
        min_value=0.0,
        value=5000.0
    )

    engagement_score = st.number_input(
        "Engagement Score",
        min_value=0.0,
        value=75.0
    )

st.markdown("### Marketing Channels")

col1, col2, col3 = st.columns(3)

with col1:
    email = st.checkbox("Email")

    facebook = st.checkbox("Facebook")

with col2:
    google = st.checkbox("Google")

    instagram = st.checkbox("Instagram")

with col3:
    whatsapp = st.checkbox("WhatsApp")

    youtube = st.checkbox("YouTube")

st.markdown("")

predict_button = st.button(
    "Predict Profit / Loss",
    use_container_width=True
)

# ==========================================================
# CLASSIFICATION PREDICTION
# ==========================================================

if predict_button:

    input_data = pd.DataFrame(
        {
            "Campaign_Type": [campaign_type],
            "Target_Audience": [target],
            "Language": [language],
            "Customer_Segment": [customer_segment],
            "Brand": [brand],
            "Duration_Category": [duration_category],
            "Duration": [duration],
            "Impressions": [impressions],
            "Clicks": [clicks],
            "Leads": [leads],
            "Conversions": [conversions],
            "Acquisition_Cost": [acquisition_cost],
            "Engagement_Score": [engagement_score],
            "Email": [email],
            "Facebook": [facebook],
            "Google": [google],
            "Instagram": [instagram],
            "WhatsApp": [whatsapp],
            "YouTube": [youtube]
        }
    )

    transformed_data = preprocessor.transform(input_data)

    prediction = model.predict(transformed_data)[0]

    st.markdown("---")

    st.success("Prediction Completed Successfully!")

    if prediction == "Profit":

        st.success("✅ Predicted Campaign Outcome : PROFIT")

    else:

        st.error("❌ Predicted Campaign Outcome : LOSS")

    st.markdown("---")

    st.subheader("Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )


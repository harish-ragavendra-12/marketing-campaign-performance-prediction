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


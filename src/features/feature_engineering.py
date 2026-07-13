import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from sklearn.preprocessing import MultiLabelBinarizer


def load_cleaned_data(file_name):

    file_path = PROCESSED_DATA_DIR / file_name

    df = pd.read_csv(file_path)

    return df

def create_profit_loss_flag(df):

    df["Profit_Loss"] = np.where(
        df["ROI"] >= 0,
        "Profit",
        "Loss"
    )

    return df

def encode_channel_used(df):

    mlb = MultiLabelBinarizer()

    encoded = mlb.fit_transform(df["Channel_Used"].str.split(", "))

    encoded_df = pd.DataFrame(encoded, columns=mlb.classes_)

    encoded_df.index = df.index

    merged_df = pd.concat([df, encoded_df],axis=1)

    merged_df = merged_df.drop(columns=["Channel_Used"])

    return merged_df

def create_duration_category(df):

    conditions = [
        df["Duration"] <= 15,
        (df["Duration"] > 15) & (df["Duration"] <= 45),
        df["Duration"] > 45
    ]

    choices = [
        "Short",
        "Medium",
        "Long",
    ]

    df["Duration_Category"] = np.select(
        conditions,
        choices,
        default="Unknown"
    )

    return df

def save_feature_engineered_dataset(df, file_name):

    output_path = PROCESSED_DATA_DIR / file_name

    df.to_csv(output_path, index=False)

    print("=" * 60)
    print("FEATURE ENGINEERED DATASET SAVED")
    print("=" * 60)
    print(f"Dataset saved successfully at:\n{output_path}")

def feature_engineering():

    feature_eng_dataset = load_cleaned_data("cleaned_marketing_campaign.csv")

    feature_eng_dataset = create_profit_loss_flag(feature_eng_dataset)

    feature_eng_dataset = encode_channel_used(feature_eng_dataset)

    feature_eng_dataset = create_duration_category(feature_eng_dataset)

    save_feature_engineered_dataset(
        feature_eng_dataset,
        "feature_engineered_marketing_campaign.csv"
    )

    return feature_eng_dataset

if __name__ == "__main__":
    feature_engineering()

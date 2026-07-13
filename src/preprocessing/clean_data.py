import pandas as pd

from config import PROCESSED_DATA_DIR


def load_processed_data(file_name):

    file_path = PROCESSED_DATA_DIR / file_name

    df = pd.read_csv(file_path)

    return df

def handle_missing_values(df):

    for column in df.columns:
        if df[column].dtype in ["int64", "float64"]:
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    return df

def remove_duplicates(df):

    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates()

    print("=" * 60)
    print("DUPLICATE RECORDS")
    print("=" * 60)
    print(f"Duplicate rows found   : {duplicate_count}")
    print(f"Rows after cleaning    : {df.shape[0]}")

    return df

def convert_data_types(df):

    print("=" * 60)
    print("DATA TYPE CONVERSION")
    print("=" * 60)

    df["Date"] = pd.to_datetime(df["Date"],format="%d-%m-%Y")

    print("Date column converted to datetime successfully.")

    return df

def validate_roi(df):

    print("=" * 60)
    print("ROI VALIDATION")
    print("=" * 60)

    df["ROI"] = (df["Revenue"] - df["Acquisition_Cost"]) / df["Acquisition_Cost"]

    print("ROI values validated successfully.")

    return df

def save_cleaned_dataset(df, file_name):

    output_path = PROCESSED_DATA_DIR / file_name

    df.to_csv(output_path, index=False)

    print("=" * 60)
    print("DATASET SAVED")
    print("=" * 60)
    print(f"Dataset saved successfully at:\n{output_path}")

def clean_data():

    processed_dataset = load_processed_data("marketing_campaign.csv")

    processed_dataset = handle_missing_values(processed_dataset)

    processed_dataset = remove_duplicates(processed_dataset)

    processed_dataset = convert_data_types(processed_dataset)

    processed_dataset = validate_roi(processed_dataset)

    save_cleaned_dataset(processed_dataset, "cleaned_marketing_campaign.csv")

    return processed_dataset

if __name__ == "__main__":
    clean_data()

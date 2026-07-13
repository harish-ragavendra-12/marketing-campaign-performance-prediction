import pandas as pd
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_dataset(file_path, brand):

    df = pd.read_csv(file_path)

    df["Brand"] = brand

    return df

def merge_datasets(df1, df2, df3):

    merged_df = pd.concat([df1, df2, df3], ignore_index=True)

    return merged_df

def display_dataset_info(merged_df):

    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(merged_df.shape)

    print("\n" + "=" * 60)
    print("FIRST 5 RECORDS")
    print("=" * 60)
    print(merged_df.head())

    print("\n" + "=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(merged_df.columns)

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(merged_df.dtypes)

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    merged_df.info()

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(merged_df.isnull().sum())

    print("\n" + "=" * 60)
    print("DUPLICATE RECORDS")
    print("=" * 60)
    print(f"Duplicate Rows : {merged_df.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(merged_df.describe())

def save_dataset(merged_df, file_name):

    output_path = PROCESSED_DATA_DIR / file_name

    merged_df.to_csv(output_path, index=False)

def load_data():

    nykaa_df = load_dataset(
        RAW_DATA_DIR / "nykaa_campaign_data_with_nulls.csv",
        "Nykaa"
    )

    purplle_df = load_dataset(
        RAW_DATA_DIR / "purplle_campaign_data_with_nulls.csv",
        "Purplle"
    )

    tira_df = load_dataset(
        RAW_DATA_DIR / "tira_campaign_data_with_nulls.csv",
        "Tira"
    )

    merged_df = merge_datasets(nykaa_df, purplle_df, tira_df)

    display_dataset_info(merged_df)

    save_dataset(merged_df, "marketing_campaign.csv")

    return merged_df

if __name__ == "__main__":
    load_data()

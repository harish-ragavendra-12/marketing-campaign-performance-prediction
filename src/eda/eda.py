import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import PROCESSED_DATA_DIR, FIGURES_DIR


def load_feature_engineered_data(file_name):

    file_path = PROCESSED_DATA_DIR / file_name

    df = pd.read_csv(file_path)

    return df

def dataset_overview(df):

    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("FIRST 5 RECORDS")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(df.columns)

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    df.info()

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("DUPLICATE RECORDS")
    print("=" * 60)
    print(f"Duplicate Rows : {df.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(df.describe(include="all"))

def univariate_analysis(df, column):

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[column],
        bins=30,
        edgecolor = "black"
    )

    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{column}_histogram.png"
    )

    plt.show()

    plt.close()

def box_plot_analysis(df, column):

    plt.figure(figsize=(8, 6))

    plt.boxplot(df[column])

    plt.title(f"{column} Box Plot")
    plt.ylabel(column)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{column}_boxplot.png"
    )

    plt.show()

    plt.close()

def numerical_univariate_analysis(df):

    numeric_columns = [
        "Revenue",
        "ROI",
        "Clicks",
        "Impressions",
        "Conversions",
        "Duration",
        "Engagement_Score"
    ]

    for column in numeric_columns:
        univariate_analysis(df, column)
        box_plot_analysis(df, column)

def categorical_analysis(df, column):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=column
    )

    plt.title(f"{column} Count")

    plt.xlabel(column)

    plt.ylabel("Count")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{column}_countplot.png"
    )

    plt.show()

    plt.close()

def categorical_univariate_analysis(df):

    categorical_columns = [
        "Brand",
        "Campaign_Type",
        "Target_Audience",
        "Language",
        "Customer_Segment",
        "Profit_Loss",
        "Duration_Category"
    ]

    for column in categorical_columns:
        categorical_analysis(df, column)

def scatter_plot_analysis(df, x_column, y_column):

    plt.figure(figsize=(8, 5))

    plt.scatter(df[x_column], df[y_column])

    plt.title(f"{x_column} vs {y_column}")

    plt.xlabel(x_column)
    plt.ylabel(y_column)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{x_column}_vs_{y_column}_scatterplot.png"
    )

    plt.show()

    plt.close()

def bivariate_analysis(df):

    pairs = [
        ("Revenue", "ROI"),
        ("Clicks", "Conversions"),
        ("Impressions", "Clicks"),
        ("Acquisition_Cost", "Revenue"),
        ("Duration", "Revenue")
    ]

    for x_column, y_column in pairs:
        scatter_plot_analysis(df, x_column, y_column)

def correlation_analysis(df):

    numeric_df = df.select_dtypes(include=["number"])

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "correlation_heatmap.png"
    )

    plt.show()

    plt.close()

def brand_performance_analysis(df):

    brand_summary = (
        df.groupby("Brand")[
            [
                "Revenue",
                "ROI",
                "Clicks",
                "Conversions",
                "Engagement_Score"
            ]
        ].mean().reset_index()
    )

    print("=" * 60)
    print("BRAND PERFORMANCE SUMMARY")
    print("=" * 60)

    print(brand_summary)

    metrics = [
        "Revenue",
        "ROI",
        "Clicks",
        "Conversions",
        "Engagement_Score"
    ]

    for metric in metrics:

        plt.figure(figsize=(8, 5))

        plt.bar(
            brand_summary["Brand"],
            brand_summary[metric],
            edgecolor="black"
        )

        plt.title(f"Brand vs {metric}")

        plt.xlabel("Brand")

        plt.ylabel(metric)

        plt.xticks(rotation=15)

        plt.grid(axis="y")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"brand_{metric}.png"
        )

        plt.show()

        plt.close()

def campaign_performance_analysis(df):

    campaign_summary = (
        df.groupby("Campaign_Type")[
            [
                "Revenue",
                "ROI",
                "Clicks",
                "Conversions",
                "Engagement_Score"
            ]
        ].mean().reset_index()
    )

    print("=" * 60)
    print("CAMPAIGN PERFORMANCE SUMMARY")
    print("=" * 60)

    print(campaign_summary)

    metrics = [
        "Revenue",
        "ROI",
        "Clicks",
        "Conversions",
        "Engagement_Score"
    ]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        plt.bar(
            campaign_summary["Campaign_Type"],
            campaign_summary[metric],
            edgecolor="black"
        )

        plt.title(f"Campaign Type vs {metric}")

        plt.xlabel("Campaign Type")

        plt.ylabel(metric)

        plt.xticks(rotation=15)

        plt.grid(axis="y")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"campaign_{metric}.png"
        )

        plt.show()

        plt.close()

def generate_insights(df):

    print("=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)

    brand_summary = (
        df.groupby("Brand")[
            ["Revenue", "ROI"]
        ]
        .mean()
        .reset_index()
    )

    # Highest Revenue Brand
    highest_index = brand_summary["Revenue"].idxmax()
    highest_brand = brand_summary.loc[highest_index]

    print(f"Highest Average Revenue Brand : {highest_brand['Brand']}")
    print(f"Highest Average Revenue       : {highest_brand['Revenue']:.2f}")

    # Highest ROI Brand
    highest_roi_index = brand_summary["ROI"].idxmax()
    highest_roi_brand = brand_summary.loc[highest_roi_index]

    print(f"Highest Average ROI Brand     : {highest_roi_brand['Brand']}")
    print(f"Highest Average ROI           : {highest_roi_brand['ROI']:.2f}")


    print("\n" + "=" * 60)
    print("CAMPAIGN INSIGHTS")
    print("=" * 60)

    campaign_summary = (
        df.groupby("Campaign_Type")[
            ["Revenue", "ROI"]
        ]
        .mean()
        .reset_index()
    )

    # Highest Revenue Campaign
    highest_campaign_revenue_index = campaign_summary["Revenue"].idxmax()
    highest_campaign_revenue = campaign_summary.loc[highest_campaign_revenue_index]

    print(f"Highest Average Revenue Campaign : {highest_campaign_revenue['Campaign_Type']}")
    print(f"Highest Average Revenue          : {highest_campaign_revenue['Revenue']:.2f}")

    # Highest ROI Campaign
    highest_campaign_roi_index = campaign_summary["ROI"].idxmax()
    highest_campaign_roi = campaign_summary.loc[highest_campaign_roi_index]

    print(f"Highest Average ROI Campaign     : {highest_campaign_roi['Campaign_Type']}")
    print(f"Highest Average ROI              : {highest_campaign_roi['ROI']:.2f}")


    print("\n" + "=" * 60)
    print("OVERALL DATASET INSIGHTS")
    print("=" * 60)

    average_revenue = df["Revenue"].mean()

    average_roi = df["ROI"].mean()

    average_engagement = df["Engagement_Score"].mean()

    profit_count = (df["Profit_Loss"] == "Profit").sum()

    loss_count = (df["Profit_Loss"] == "Loss").sum()

    print(f"Average Revenue               : {average_revenue:.2f}")
    print(f"Average ROI                   : {average_roi:.2f}")
    print(f"Average Engagement Score      : {average_engagement:.2f}")

    print(f"Profit Campaigns              : {profit_count}")
    print(f"Loss Campaigns                : {loss_count}")

def perform_eda():

    df = load_feature_engineered_data(
        "feature_engineered_marketing_campaign.csv"
    )

    dataset_overview(df)

    numerical_univariate_analysis(df)

    categorical_univariate_analysis(df)

    bivariate_analysis(df)

    correlation_analysis(df)

    brand_performance_analysis(df)

    campaign_performance_analysis(df)

    generate_insights(df)

    return df

if __name__ == "__main__":
    perform_eda()

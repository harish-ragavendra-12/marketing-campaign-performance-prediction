import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)


from config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR


def load_feature_engineered_data(file_name):

    file_path = PROCESSED_DATA_DIR / file_name

    df = pd.read_csv(file_path)

    return df

def prepare_features_target(df):

    X = df.drop(
        columns=[
            "Campaign_ID",
            "Revenue",
            "ROI",
            "Profit_Loss"
        ]
    )

    y = df["Revenue"]

    return X, y

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
         X,
                y,
                test_size = 0.2,
                random_state = 42
    )

    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):

    categorical_columns = [
        "Campaign_Type",
        "Target_Audience",
        "Language",
        "Customer_Segment",
        "Brand",
        "Duration_Category"
    ]

    numerical_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(),
                categorical_columns
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_columns
            )
        ]
    )

    X_train = preprocessor.fit_transform(X_train)

    X_test = preprocessor.transform(X_test)

    return X_train, X_test, preprocessor

def train_linear_regression(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model

def train_decision_tree(X_train, y_train):

    model = DecisionTreeRegressor()

    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    return model

def evaluate_model(model_name, model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = root_mean_squared_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print("=" * 60)
    print(f"{model_name.upper()} EVALUATION")
    print("=" * 60)

    print(f"Mean Absolute Error     : {mae:.2f}")
    print(f"Root Mean Squared Error : {rmse:.2f}")
    print(f"R2 Score                : {r2:.4f}")

    return mae, rmse, r2

def regression_pipeline():

    df = load_feature_engineered_data("feature_engineered_marketing_campaign.csv")

    X, y = prepare_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, preprocessor = preprocess_data(X_train, X_test)

    linear_model = train_linear_regression(X_train, y_train)

    decision_tree_model = train_decision_tree(X_train, y_train)

    random_forest_model = train_random_forest(X_train, y_train)

    linear_results = evaluate_model(
        "Linear Regression",
        linear_model,
        X_test,
        y_test
    )

    decision_tree_results = evaluate_model(
        "Decision Tree",
        decision_tree_model,
        X_test,
        y_test
    )

    random_forest_results = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test
    )

    comparison_df, best_model = compare_models(
        linear_results,
        decision_tree_results,
        random_forest_results
    )

    save_best_model(
        best_model,
        linear_model,
        decision_tree_model,
        random_forest_model,
        preprocessor
    )

    return comparison_df, best_model

def compare_models(linear_results, decision_tree_results, random_forest_results):

    comparison_df = pd.DataFrame(
        [
            [
                "Linear Regression",
                linear_results[0],
                linear_results[1],
                linear_results[2]
            ],
            [
                "Decision Tree Regressor",
                decision_tree_results[0],
                decision_tree_results[1],
                decision_tree_results[2]
            ],
            [
                "Random Forest Regressor",
                random_forest_results[0],
                random_forest_results[1],
                random_forest_results[2]
            ]
        ],
        columns=[
            "Model",
            "MAE",
            "RMSE",
            "R2 Score"
        ]
    )

    comparison_df.to_csv(
        REPORTS_DIR / "regression_results.csv",
        index=False
    )

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(comparison_df)


    best_model_index = comparison_df["R2 Score"].idxmax()

    best_model = comparison_df.loc[best_model_index]

    print("\n" + "=" * 60)
    print("BEST REGRESSION MODEL")
    print("=" * 60)

    print(f"Model     : {best_model['Model']}")
    print(f"MAE       : {best_model['MAE']:.2f}")
    print(f"RMSE      : {best_model['RMSE']:.2f}")
    print(f"R2 Score  : {best_model['R2 Score']:.4f}")

    return comparison_df, best_model

def save_best_model(
        best_model,
        linear_model,
        decision_tree_model,
        random_forest_model,
        preprocessor
):

    if best_model["Model"] == "Linear Regression":
        joblib.dump(linear_model, MODELS_DIR / "best_regression_model.pkl")

    elif best_model["Model"] == "Decision Tree Regressor":
        joblib.dump(decision_tree_model, MODELS_DIR / "best_regression_model.pkl")

    else:
        joblib.dump(random_forest_model, MODELS_DIR / "best_regression_model.pkl")

    # Save the fitted preprocessor
    joblib.dump(preprocessor, MODELS_DIR / "regression_preprocessor.pkl")

    print("=" * 60)
    print("BEST MODEL SAVED")
    print("=" * 60)

    print(f"Saved Model        : {best_model['Model']}")
    print(f"Location           : {MODELS_DIR / 'best_regression_model.pkl'}")
    print(f"Preprocessor Saved : {MODELS_DIR / 'regression_preprocessor.pkl'}")

if __name__ == "__main__":
    regression_pipeline()
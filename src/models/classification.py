import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
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

    y = df["Profit_Loss"]

    return X, y

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
         X,
                y,
                test_size = 0.2,
                random_state = 42,
                stratify=y
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

def train_logistic_regression(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model

def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    return model

def evaluate_model(model_name, model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="Profit"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="Profit"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Profit"
    )

    print("=" * 60)
    print(f"{model_name.upper()} EVALUATION")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return accuracy, precision, recall, f1

def classification_pipeline():

    df = load_feature_engineered_data("feature_engineered_marketing_campaign.csv")

    X, y = prepare_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, preprocessor = preprocess_data(X_train, X_test)

    logistic_model = train_logistic_regression(X_train, y_train)

    decision_tree_model = train_decision_tree(X_train, y_train)

    random_forest_model = train_random_forest(X_train, y_train)

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_model,
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
        logistic_results,
        decision_tree_results,
        random_forest_results
    )

    save_best_model(
        best_model,
        logistic_model,
        decision_tree_model,
        random_forest_model,
        preprocessor
    )

    return comparison_df, best_model

def compare_models(logistic_results, decision_tree_results, random_forest_results):

    comparison_df = pd.DataFrame(
        [
            [
                "Logistic Regression",
                logistic_results[0],
                logistic_results[1],
                logistic_results[2],
                logistic_results[3]
            ],
            [
                "Decision Tree Classifier",
                decision_tree_results[0],
                decision_tree_results[1],
                decision_tree_results[2],
                decision_tree_results[3]
            ],
            [
                "Random Forest Classifier",
                random_forest_results[0],
                random_forest_results[1],
                random_forest_results[2],
                random_forest_results[3]
            ]
        ],
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    )

    comparison_df.to_csv(
        REPORTS_DIR / "classification_results.csv",
        index=False
    )

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(comparison_df)


    best_model_index = comparison_df["F1 Score"].idxmax()

    best_model = comparison_df.loc[best_model_index]

    print("\n" + "=" * 60)
    print("BEST CLASSIFICATION MODEL")
    print("=" * 60)

    print(f"Model     : {best_model['Model']}")
    print(f"Accuracy  : {best_model['Accuracy']:.4f}")
    print(f"Precision : {best_model['Precision']:.4f}")
    print(f"Recall    : {best_model['Recall']:.4f}")
    print(f"F1 Score  : {best_model['F1 Score']:.4f}")

    return comparison_df, best_model

def save_best_model(
        best_model,
        logistic_model,
        decision_tree_model,
        random_forest_model,
        preprocessor
):

    if best_model["Model"] == "Logistic Regression":
        joblib.dump(logistic_model, MODELS_DIR / "best_classification_model.pkl")

    elif best_model["Model"] == "Decision Tree Classifier":
        joblib.dump(decision_tree_model, MODELS_DIR / "best_classification_model.pkl")

    else:
        joblib.dump(random_forest_model, MODELS_DIR / "best_classification_model.pkl")


    joblib.dump(preprocessor, MODELS_DIR / "classification_preprocessor.pkl")

    print("=" * 60)
    print("BEST MODEL SAVED")
    print("=" * 60)

    print(f"Saved Model        : {best_model['Model']}")
    print(f"Location           : {MODELS_DIR / 'best_classification_model.pkl'}")
    print(f"Preprocessor Saved : {MODELS_DIR / 'classification_preprocessor.pkl'}")

if __name__ == "__main__":
    classification_pipeline()
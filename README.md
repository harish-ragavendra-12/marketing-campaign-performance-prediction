# 📈 Marketing Campaign Performance Prediction

An end-to-end **Data Science and Machine Learning** project that analyzes multi-brand marketing campaign performance and predicts **campaign revenue** and **campaign profitability** using Regression and Classification models. The project also includes an interactive **Streamlit Dashboard** for data exploration, predictions, and business insights.

---

## 📌 Project Overview

Marketing campaigns generate large volumes of data across multiple brands and channels. Analyzing this data helps businesses understand campaign effectiveness, optimize marketing budgets, and improve return on investment (ROI).

This project performs:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Regression Modeling (Revenue Prediction)
- Classification Modeling (Profit/Loss Prediction)
- Model Comparison
- Interactive Streamlit Dashboard
- Business Insights Generation

---

## 🎯 Business Objectives

- Analyze marketing campaign performance.
- Identify high-performing brands and campaign types.
- Predict campaign revenue using Machine Learning.
- Predict campaign profitability (Profit/Loss).
- Compare multiple Machine Learning models.
- Generate actionable business insights.
- Provide an interactive dashboard for decision-making.

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

### Development Tools
- PyCharm
- Git
- GitHub

---

## 📂 Project Structure

```
marketing_campaign_prediction/
│
├── app.py
├── config.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│       └── feature_engineered_marketing_campaign.csv
│
├── models/
│   ├── best_regression_model.pkl
│   ├── regression_preprocessor.pkl
│   ├── best_classification_model.pkl
│   └── classification_preprocessor.pkl
│
├── reports/
│   ├── figures/
│   ├── regression_results.csv
│   └── classification_results.csv
│
├── pages/
│   ├── 1_Home.py
│   ├── 2_Dataset_Overview.py
│   ├── 3_EDA.py
│   ├── 4_Regression_Prediction.py
│   ├── 5_Classification_Prediction.py
│   ├── 6_Model_Comparison.py
│   ├── 7_Business_Insights.py
│   └── 8_About.py
│
└── src/
    ├── data/
    ├── preprocessing/
    ├── features/
    ├── eda/
    └── models/
```

---

## 📊 Dataset Information

The project uses marketing campaign datasets from three brands:

- Nykaa
- Purplle
- Tira

The datasets include campaign performance metrics such as:

- Campaign Type
- Target Audience
- Language
- Customer Segment
- Brand
- Duration
- Impressions
- Clicks
- Leads
- Conversions
- Acquisition Cost
- Engagement Score
- Marketing Channels
- Revenue
- ROI

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Removing duplicate records
- Data type conversion
- ROI validation
- Feature engineering
- Multi-label encoding for marketing channels
- Duration categorization

---

## ⚡ Feature Engineering

New features created include:

- Profit/Loss Flag
- ROI Validation
- Duration Category
- One-Hot Encoded Marketing Channels

---

## 📈 Exploratory Data Analysis

Performed analysis on:

- Revenue Distribution
- ROI Distribution
- Campaign Type Analysis
- Brand Performance
- Customer Segment Analysis
- Correlation Analysis
- Outlier Detection
- Missing Value Analysis

---

# 🤖 Machine Learning Models

## Regression Models

Target Variable:

- Revenue

Models Trained:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation Metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Classification Models

Target Variable:

- Profit / Loss

Models Trained:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 🏆 Model Comparison

The project automatically compares all trained models and selects the best-performing model based on evaluation metrics.

Regression:

- Best model selected using **Highest R² Score**

Classification:

- Best model selected using **Highest F1 Score**

The selected models are saved using Joblib for deployment.

---

## 💾 Saved Models

Regression:

- best_regression_model.pkl
- regression_preprocessor.pkl

Classification:

- best_classification_model.pkl
- classification_preprocessor.pkl

---

# 📊 Streamlit Dashboard

The dashboard includes the following pages:

### 🏠 Home

- Project overview
- Objectives
- Technology stack
- Workflow

### 📊 Dataset Overview

- Dataset preview
- Dataset statistics
- Missing values
- Data types

### 📈 Exploratory Data Analysis

- Revenue analysis
- ROI analysis
- Brand analysis
- Campaign analysis
- Correlation heatmap

### 💰 Regression Prediction

Predict campaign revenue using the best regression model.

### 🎯 Classification Prediction

Predict whether a campaign will result in **Profit** or **Loss**.

### 📋 Model Comparison

Compare Regression and Classification models.

### 💡 Business Insights

- KPI Cards
- Brand Performance
- Campaign Performance
- Business Recommendations

### ℹ️ About

Project documentation and workflow.

---

## 🚀 Project Workflow

```
Raw Data
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Regression Modeling
     │
     ▼
Classification Modeling
     │
     ▼
Model Evaluation
     │
     ▼
Model Comparison
     │
     ▼
Model Saving
     │
     ▼
Streamlit Dashboard
```

---

## ▶️ How to Run the Project

### Clone the repository

```bash
git clone https://github.com/harish-ragavendra-12/marketing_campaign_prediction.git
```

### Navigate to the project directory

```bash
cd marketing_campaign_prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 📌 Future Enhancements

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Feature importance visualization
- SHAP model explainability
- Streamlit Cloud deployment
- Docker containerization
- CI/CD pipeline integration

---

## 👨‍💻 Developer

**Harish Ragavendra**

Data Science Portfolio Project

---

## ⭐ If you found this project useful, consider giving it a star!
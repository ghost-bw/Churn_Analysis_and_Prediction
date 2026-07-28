# 🔮 ChurnLens — Customer Churn Analysis & Prediction

> An end-to-end Data Analytics and Machine Learning project that identifies at-risk customers, predicts churn probability, and delivers actionable business retention recommendations through an interactive Streamlit web application.

Live Link- https://churn--lens.streamlit.app/
---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Live Application](#-live-application)
3. [Project Structure](#-project-structure)
4. [Dataset](#-dataset)
5. [Feature Engineering](#-feature-engineering)
6. [Model Development](#-model-development)
7. [Model Comparison & Selection](#-model-comparison--selection)
8. [Feature Importance](#-feature-importance)
9. [Streamlit Application](#-streamlit-application)
10. [Business Insights & Recommendations](#-business-insights--recommendations)
11. [Installation & Setup](#-installation--setup)
12. [Usage Guide](#-usage-guide)
13. [Tech Stack](#-tech-stack)
14. [Author](#-author)

---

## 🎯 Project Overview

Customer churn — the loss of a paying customer — is one of the most costly problems in subscription-based businesses. This project builds a **complete analytics pipeline** that:

- Extracts and merges customer data from a **SQLite relational database**
- Performs **exploratory data analysis (EDA)** to surface churn-driving patterns
- Engineers derived features from raw date fields (customer age, subscription tenure, days since last complaint)
- Trains and compares **4 machine learning classifiers** (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Selects the best model based on **ROC-AUC** and **Recall** (prioritizing catching churners over minimizing false alarms)
- Deploys a **production-ready Streamlit web app** (ChurnLens) for individual and batch churn prediction
- Generates **data-driven business recommendations** tailored to each customer's risk level

---

## 🚀 Live Application

Run the Streamlit app locally:

```bash
streamlit run app/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
Churn_Analysis/
│
├── app/
│   └── app.py                          # Main Streamlit application (ChurnLens)
│
├── data/
│   ├── customer_churn.db               # SQLite database with raw relational tables
│   ├── exported_churn_data.csv         # Raw merged dataset exported from DB (521 rows × 21 cols)
│   └── cleaned_churn_data.csv          # Feature-engineered, model-ready dataset
│
├── models/
│   ├── logistic_regression_best_model.joblib   # Serialized pipeline (preferred)
│   └── logistic_regression_best_model.pkl      # Serialized pipeline (backup)
│
├── notebooks/
│   ├── churn.ipynb                     # Business Analytics notebook (EDA, SQL, visualizations)
│   ├── Churn_Prediction.ipynb          # Machine Learning notebook (training, evaluation, export)
│   └── PowerBI_Insights.ipynb          # Power BI data prep helper notebook
│
├── reports/
│   ├── classification_report.txt       # Full classification reports for all 4 models
│   ├── model_comparison.csv            # Side-by-side model metric comparison table
│   └── feature_importance.csv          # Logistic Regression coefficients (35 features)
│
├── dashboard/
│   └── customer_churn.pbix             # Power BI dashboard file
│
├── requirements.txt                    # Python dependencies
├── info.txt                            # Auto-generated model & dataset diagnostic info
└── README.md                           # This file
```

---

## 📊 Dataset

The raw data originates from a **SQLite relational database** (`data/customer_churn.db`) and is merged into a single flat dataset of **521 customers × 21 columns**.

### Key Columns

| Column | Type | Description |
|---|---|---|
| `customerid` | String | Unique customer identifier |
| `customer_name` | String | Customer full name |
| `gender` | Categorical | Male / Female |
| `dob` | Date | Date of birth (used to derive `customer_age`) |
| `country` | Categorical | India / Nepal |
| `state` | Categorical | 12 unique states/regions |
| `subscription_start_date` | Date | When subscription began |
| `renewal_date` | Date | Subscription renewal date |
| `subscription_type` | Categorical | Refferal / Paid / Organic |
| `plan_type` | Categorical | Standard / Premium / Basic |
| `contract_type` | Categorical | Annual / Monthly |
| `monthly_charges` | Numeric | Recurring billing amount ($6.99 – $92.99) |
| `cltv` | Numeric | Customer Lifetime Value (31 – 2185) |
| `churn_score` | Numeric | Internal risk score (3 – 99) |
| `csat_score` | Numeric | Customer satisfaction score (10 – 95) |
| `complaint_count` | Numeric | Number of complaints raised (1 – 3) |
| `complaint_date` | Date | Date of last complaint |
| `escalations` | Categorical | Whether complaint was escalated (Y / N) |
| `cancellation_reason` | String | Reason for cancellation (if churned) |
| **`churn_flag`** | Binary Target | **0 = Retained, 1 = Churned** |

---

## 🔧 Feature Engineering

Three derived features are calculated from date columns:

| Derived Feature | Formula | Purpose |
|---|---|---|
| `customer_age` | `(today - dob).days / 365.25` | Customer's current age in years |
| `subscription_duration_months` | `(renewal_date - subscription_start_date).days / 30.44` | Tenure in months |
| `days_since_last_complaint` | `(today - complaint_date).days` | Recency of support issues |

---

## 🤖 Model Development

### Preprocessing Pipeline

A **scikit-learn `Pipeline`** with a `ColumnTransformer` handles:

- **Numerical features (8)**: `StandardScaler` + `SimpleImputer(strategy='median')`
- **Categorical features (7)**: `OneHotEncoder` + `SimpleImputer(strategy='most_frequent')`

### Class Imbalance Handling

The dataset contains a minority churn class. The Logistic Regression model uses **`class_weight='balanced'`** to upweight churned customers, maximizing recall on the positive class (actual churners).

### Models Trained

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **71.4%** | **50.0%** | **76.7%** | **60.5%** | **81.3%** |
| XGBoost | 78.1% | 65.2% | 50.0% | 56.6% | 73.1% |
| Random Forest | 73.3% | 55.0% | 36.7% | 44.0% | 72.8% |
| Decision Tree | 65.7% | 40.0% | 40.0% | 40.0% | 58.0% |

---

## 🏆 Model Comparison & Selection

**Logistic Regression** was selected as the production model despite XGBoost having higher accuracy, for the following reasons:

1. **Highest ROC-AUC (81.3%)**: Best overall discrimination between churners and retained customers across all classification thresholds.

2. **Highest Recall on Churn Class (76.7%)**: In a churn use case, **missing a churner is far more costly** than a false alarm. Logistic Regression catches 77 out of every 100 actual churners vs. only 50 for XGBoost.

3. **Interpretability**: Logistic Regression coefficients directly reveal which features increase or decrease churn probability — essential for communicating findings to business stakeholders.

4. **Deployment Efficiency**: Fast inference with no tree-traversal overhead, ideal for real-time and batch prediction workloads.

> 📌 **Business Rationale**: A missed churner costs the business the entire future revenue of that customer (CLTV). A false alarm costs a small retention discount. Therefore **maximizing Recall over Accuracy** is the correct optimization objective.

---

## 📈 Feature Importance

Top features by absolute logistic regression coefficient:

| Rank | Feature | Coefficient | Direction |
|---|---|---|---|
| 1 | `churn_score` | +1.326 | ↑ Increases churn risk |
| 2 | `escalations_Y` | +0.970 | ↑ Increases churn risk |
| 3 | `state_Gujarat` | +0.966 | ↑ Increases churn risk |
| 4 | `escalations_N` | −0.914 | ↓ Reduces churn risk |
| 5 | `state_Karnataka` | +0.813 | ↑ Increases churn risk |
| 6 | `state_Nagaland` | −0.631 | ↓ Reduces churn risk |
| 7 | `state_Maharashtra` | −0.611 | ↓ Reduces churn risk |
| 8 | `csat_score` | +0.194 | ↑ Increases churn risk |
| 9 | `contract_type_Monthly` | +0.078 | ↑ Increases churn risk |

**Key Findings:**
- The **internal churn score** is the single strongest churn predictor.
- **Support escalations** significantly elevate risk when present, and their absence is a protective factor.
- **Regional patterns** exist — Gujarat and Karnataka show elevated risk while Nagaland and Maharashtra show lower risk.
- **Monthly contracts** carry higher churn risk than Annual.

---

## 🖥️ Streamlit Application

The [app/app.py](app/app.py) implements **ChurnLens**, a three-page interactive web application.

### Pages

#### 🏠 Home
- Project overview and model methodology summary
- Interactive **feature importance bar chart** (Plotly)
- Dataset KPIs: Total Customers, Overall Churn Rate, Avg Monthly Charges, Avg CLTV
- Churn distribution charts by Contract Type and Plan Type
- Strategic business insights panel with 4 actionable retention strategies

#### 🔮 Predict Churn
**Tab 1 — Manual Input Prediction:**
- Input form across 3 organized columns (Personal, Billing, Support)
- Accepts all 15 model features with sensible defaults
- Outputs:
  - **Churn prediction** (Retained / Churn Warning) with color coding
  - **Churn probability** percentage
  - **Risk Level badge** (🟢 Low / 🟡 Medium / 🔴 High)
  - **Tailored business recommendations** for the predicted risk level

**Tab 2 — Batch CSV Prediction:**
- Template CSV download button for schema reference
- CSV file uploader with **automatic feature engineering** (date parsing, age/tenure calculation)
- Instant upload preview charts (charges distribution, plan type distribution)
- Batch prediction producing `churn_probability`, `churn_prediction`, `churn_risk_level`, `actionable_recommendation` columns
- Summary metrics: Total Customers, Predicted Churn Rate, High Risk Count
- 3 interactive visualizations: Risk Level Pie, Avg Churn Probability by Plan, Scatter by Monthly Charges
- Dynamic batch insights (contract attrition, escalation rates, low CSAT analysis)
- **📥 Download Complete Predictions CSV** — exports full predictions table

#### ℹ️ About
- Feature catalogue (8 numerical + 7 categorical features)
- Preprocessing methodology (StandardScaler, OneHotEncoder, SimpleImputer)

### Risk Level Logic

| Probability Range | Risk Level | Action |
|---|---|---|
| ≥ 70% | 🔴 High | Immediate outreach, financial incentive, dedicated support |
| 40% – 69% | 🟡 Medium | Personalized survey, feature walkthrough, loyalty reward |
| < 40% | 🟢 Low | Regular engagement, annual contract promotion |

---

## 💡 Business Insights & Recommendations

Based on historical patterns in the dataset:

1. **Monthly Contract Volatility** — Monthly contract holders churn at significantly higher rates than annual customers.
   - *Action*: Offer 15% discount or 1 free month to incentivize switch to Annual.

2. **Support Escalation Impact** — Escalated complaints are a leading churn indicator, especially when combined with CSAT < 60.
   - *Action*: Create a "Priority Retention Support" team. Reach out within 24 hours of any escalation.

3. **High Billing Thresholds** — Churn risk rises sharply for monthly charges above $50, particularly for short-tenure customers.
   - *Action*: Audit accounts paying over $50/month. Offer bundled features or a down-sell path to retain them.

4. **Referral Acquisition Loyalty** — Customers acquired through referrals show longer tenures and lower attrition.
   - *Action*: Expand referral programs. Offer billing credits for every successful referral from low-risk current customers.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone or Download the Project

```bash
git clone <repo-url>
cd Churn_Analysis
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app/app.py
```

The app will open automatically at [http://localhost:8501](http://localhost:8501).

---

## 📖 Usage Guide

### Running Individual Predictions

1. Navigate to **Predict Churn → Manual Input Prediction**
2. Fill in the customer's attributes across all three input columns
3. Click **"Predict Churn Risk"**
4. Review the prediction output, risk badge, customer summary, and business recommendations

### Running Batch Predictions

1. Navigate to **Predict Churn → Batch CSV Prediction**
2. Click **"📥 Download Upload Template"** to get the correct CSV schema
3. Fill in your customer data following the template structure
4. Upload the completed CSV using the file uploader
5. Click **"▶ Run Batch Prediction"** to process all records
6. Review the batch summary, charts, and actionable insights
7. Click **"📥 Download Complete Predictions CSV"** to export the results

> **Supported date columns in uploads**: `dob`, `subscription_start_date`, `renewal_date`, `complaint_date`
> These will be automatically parsed and converted to the derived model features.

### Exploring Notebooks

| Notebook | Description |
|---|---|
| `notebooks/churn.ipynb` | Full business analytics: SQL queries, EDA, visualizations, data profiling |
| `notebooks/Churn_Prediction.ipynb` | ML pipeline: preprocessing, model training, evaluation, model export |
| `notebooks/PowerBI_Insights.ipynb` | Data preparation for Power BI dashboard |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Web Application** | Streamlit |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | scikit-learn (Pipeline, LogisticRegression, ColumnTransformer) |
| **Boosting** | XGBoost |
| **Visualization** | Plotly Express, Plotly Graph Objects |
| **Model Serialization** | joblib |
| **Database** | SQLite (via Python `sqlite3`) |
| **BI Dashboard** | Microsoft Power BI |
| **Notebooks** | Jupyter Notebook |

---

## 👤 Author

**Raunak Prabhakar**

> This project was built as a complete end-to-end Data Analytics and Machine Learning portfolio project, covering the full spectrum from raw database extraction to interactive deployment.

---

*Built with ❤️ using Python & Streamlit*

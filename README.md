# 💳 Loanify — Intelligent Loan Approval & Credit Risk Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4%2B-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Loanify** is a production-ready Machine Learning web application and REST API that automates loan eligibility evaluation and borrower risk assessment. Powered by a trained **Gaussian Naive Bayes** classifier and a **27-feature StandardScaler** pipeline, Loanify delivers instant approval decisions, calibrated confidence scores, and explainable financial health breakdowns.

---

## 🌟 Key Features

* **⚡ Modern Responsive Web UI**: Built with Tailwind CSS, Lucide icons, and sleek dark-mode glassmorphism.
* **🧠 Machine Learning Engine**: Calibrated `GaussianNB` classifier paired with fitted `StandardScaler` (`credit_wise_model.pkl` & `credit_wise_scaler.pkl`).
* **📐 Non-Linear Feature Engineering**: Includes quadratic interaction terms ($\text{Credit\_Score}^2$ and $\text{DTI\_Ratio}^2$) for sharper risk separation.
* **📊 Comprehensive Risk Insights**:
  * Approval & Rejection probabilities (e.g. $98.1\%$ vs $1.9\%$).
  * Automated Risk Tier (`Low Risk`, `Moderate Risk`, `High Risk`).
  * Debt-to-Income (DTI) health and collateral coverage ratios.
  * Personalized advisory recommendations and key risk flags.
* **📂 Batch CSV Scoring**: Upload an applicant spreadsheet in Streamlit and download evaluated predictions with probabilities.
* **🔌 REST API & Swagger Docs**: Fully documented OpenAPI 3.0 endpoints (`/docs` and `/redoc`).
* **⚡ 1-Click Launchers**: Double-click `.bat` scripts for Windows to start FastAPI or Streamlit immediately.

---

## 📁 Repository Structure

```text
Loanify/
├── app.py                     # FastAPI backend + Integrated responsive dashboard
├── predictor.py               # Core inference engine & 27-feature pipeline
├── streamlit_app.py           # Streamlit dashboard & Batch CSV scoring
├── credit_wise_model.pkl      # Trained Gaussian Naive Bayes Model
├── credit_wise_scaler.pkl     # Fitted 27-Feature StandardScaler
├── requirements.txt           # Project dependencies
├── run.py                     # Interactive CLI launcher
├── start_server.bat           # 1-Click launcher for FastAPI server
├── start_streamlit.bat        # 1-Click launcher for Streamlit UI
├── frontend/                  # Decoupled standalone frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── dataset/                   # Training dataset
│   └── loan_approval_data.csv
├── notebooks/                 # Model training & EDA notebook
│   └── credit_wise_model_training.ipynb
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/amanpandeyyy/Loanify.git
cd Loanify

pip install -r requirements.txt
```

---

### 2. Run the Application

#### Option A: FastAPI Backend & Full Web Dashboard (Recommended)
```bash
python app.py
```
* **Web UI Dashboard**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive API Docs (Swagger)**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### Option B: Streamlit Data Science Dashboard
```bash
streamlit run streamlit_app.py
```
* **Streamlit UI**: Open [http://localhost:8501](http://localhost:8501)
* Includes single applicant scoring + **Batch CSV Upload & Scoring**.

#### Option C: Windows 1-Click Launchers
Simply double-click:
* `start_server.bat` (Starts FastAPI and web app)
* `start_streamlit.bat` (Starts Streamlit dashboard)

---

## 🔬 Machine Learning Pipeline & 27 Features

The model processes **27 standardized features**:

| Index | Feature | Description |
| :---: | :--- | :--- |
| **0** | `Applicant_Income` | Monthly income in USD |
| **1** | `Coapplicant_Income` | Co-applicant monthly income in USD |
| **2** | `Age` | Applicant age |
| **3** | `Dependents` | Number of financial dependents |
| **4** | `Existing_Loans` | Count of active credit accounts |
| **5** | `Savings` | Total liquid savings balance |
| **6** | `Collateral_Value` | Appraised market value of pledged collateral |
| **7** | `Loan_Amount` | Principal loan amount requested |
| **8** | `Loan_Term` | Loan duration in months |
| **9** | `Education_Level` | `Graduate` (0) or `Not Graduate` (1) |
| **10–12** | `Employment_Status` | One-hot: Salaried, Self-employed, Unemployed |
| **13** | `Marital_Status` | Single (1) or Married (0) |
| **14–17** | `Loan_Purpose` | One-hot: Car, Education, Home, Personal |
| **18–19** | `Property_Area` | One-hot: Semiurban, Urban |
| **20** | `Gender` | Male (1) or Female (0) |
| **21–24** | `Employer_Category`| One-hot: Government, MNC, Private, Unemployed |
| **25** | `DTI_Ratio_sq` | Non-linear interaction: $\text{DTI\_Ratio}^2$ |
| **26** | `Credit_Score_sq` | Non-linear interaction: $\text{Credit\_Score}^2$ |

---

## 📡 REST API Reference

### Healthcheck
`GET /api/health`
```json
{
  "status": "healthy",
  "model_type": "GaussianNB",
  "scaler_type": "StandardScaler",
  "feature_count": 27,
  "classes": [0, 1]
}
```

### Pre-loaded Test Scenarios
`GET /api/scenarios`
Returns pre-configured profiles for **Prime Approved**, **Moderate Risk**, and **High Risk (Reject)**.

### Predict Loan Approval
`POST /api/predict`

**Sample Request Body:**
```json
{
  "Applicant_Income": 14500.0,
  "Coapplicant_Income": 4000.0,
  "Age": 36.0,
  "Dependents": 1.0,
  "Existing_Loans": 1.0,
  "Savings": 16000.0,
  "Collateral_Value": 38000.0,
  "Loan_Amount": 20000.0,
  "Loan_Term": 48.0,
  "Credit_Score": 735.0,
  "DTI_Ratio": 0.25,
  "Education_Level": "Graduate",
  "Employment_Status": "Salaried",
  "Marital_Status": "Married",
  "Loan_Purpose": "Home",
  "Property_Area": "Urban",
  "Gender": "Male",
  "Employer_Category": "MNC"
}
```

**Sample Response Body:**
```json
{
  "success": true,
  "data": {
    "loan_approved": true,
    "decision": "APPROVED",
    "approval_probability": 98.1,
    "rejection_probability": 1.9,
    "risk_tier": "Low Risk",
    "metrics": {
      "credit_score": 735,
      "credit_score_tier": "Good",
      "dti_ratio": 0.25,
      "dti_status": "Healthy (<30%)",
      "collateral_coverage_ratio": 1.9,
      "income_to_loan_ratio": 0.93,
      "total_monthly_income": 18500.0
    },
    "insights": {
      "strengths": [
        "Strong credit score (735) indicates high repayment reliability.",
        "Low Debt-to-Income ratio (0.25) indicates manageable monthly obligations.",
        "Solid collateral coverage (1.9x requested loan amount).",
        "Healthy emergency savings reserve ($16,000)."
      ],
      "risks": [],
      "recommendations": [
        "Maintain timely payments and avoid taking on new short-term debt."
      ]
    }
  }
}
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

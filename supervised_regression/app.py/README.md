# CreditWise AI — Loan Approval Prediction System

An end-to-end Machine Learning web application and API that predicts loan eligibility and assesses borrower risk using a trained **Gaussian Naive Bayes** classifier (`credit_wise_model.pkl`) and calibrated **StandardScaler** (`credit_wise_scaler.pkl`).

---

## 📁 Project Structure

```text
supervised_regression/
└── app.py/
    ├── credit_wise_model.pkl      # Trained Gaussian Naive Bayes Model
    ├── credit_wise_scaler.pkl     # Fitted 27-Feature StandardScaler
    ├── predictor.py               # Core inference engine & feature transformation
    ├── app.py                     # FastAPI backend + Integrated responsive UI
    ├── streamlit_app.py           # Streamlit dashboard with batch CSV scoring
    ├── run.py                     # Interactive CLI launcher
    ├── frontend/                  # Standalone decoupled frontend
    │   ├── index.html
    │   ├── style.css
    │   └── app.js
    └── README.md
```

---

## 🚀 How to Run

### Option 1: FastAPI Backend + Integrated Web Dashboard (Recommended)
Run with your Python interpreter:
```bash
python app.py
```
Or with Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
- **Web App**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive Swagger API Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### Option 2: Streamlit Interactive ML App & Batch CSV Scoring
```bash
streamlit run streamlit_app.py
```
- Open [http://localhost:8501](http://localhost:8501).
- Includes **Single Applicant Evaluation** with instant probability gauges and **Batch CSV Upload & Scoring**.

---

### Option 3: Interactive Launcher
```bash
python run.py
```
Prompts you to start either the FastAPI application or the Streamlit dashboard with a single keypress.

---

### Option 4: Standalone HTML/CSS/JS Frontend
If you are running the backend on `http://127.0.0.1:8000`, you can open `frontend/index.html` directly in any web browser or host it on GitHub Pages / Vercel.

---

## 🧠 ML Features & Preprocessing Pipeline

The system expects 27 standardized inputs derived from user entries:
1. `Applicant_Income` (Monthly income in USD)
2. `Coapplicant_Income` (Monthly co-applicant income in USD)
3. `Age` (Years)
4. `Dependents` (Count)
5. `Existing_Loans` (Active credit lines count)
6. `Savings` (Liquid savings balance in USD)
7. `Collateral_Value` (Appraised collateral in USD)
8. `Loan_Amount` (Requested loan principal)
9. `Loan_Term` (Duration in months)
10. `Education_Level` (`Graduate` = 0, `Not Graduate` = 1)
11–13. `Employment_Status` (Salaried, Self-employed, Unemployed)
14. `Marital_Status` (Single = 1, Married = 0)
15–18. `Loan_Purpose` (Car, Education, Home, Personal)
19–20. `Property_Area` (Semiurban, Urban)
21. `Gender` (Male = 1, Female = 0)
22–24. `Employer_Category` (Government, MNC, Private, Unemployed)
25. `DTI_Ratio_sq` ($\text{DTI\_Ratio}^2$)
26. `Credit_Score_sq` ($\text{Credit\_Score}^2$)

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

### Pre-configured Sample Scenarios
`GET /api/scenarios`

### Predict Loan Approval
`POST /api/predict`
**Request Body:**
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
**Response Body:**
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

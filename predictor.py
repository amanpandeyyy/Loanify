import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Exact 27 features expected by credit_wise_scaler.pkl
FEATURE_NAMES = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Existing_Loans",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "Education_Level",
    "Employment_Status_Salaried",
    "Employment_Status_Self-employed",
    "Employment_Status_Unemployed",
    "Marital_Status_Single",
    "Loan_Purpose_Car",
    "Loan_Purpose_Education",
    "Loan_Purpose_Home",
    "Loan_Purpose_Personal",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
    "Gender_Male",
    "Employer_Category_Government",
    "Employer_Category_MNC",
    "Employer_Category_Private",
    "Employer_Category_Unemployed",
    "DTI_Ratio_sq",
    "Credit_Score_sq"
]


class CreditWisePredictor:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Candidate locations for model and scaler
        candidate_model_paths = [
            model_path,
            os.path.join(base_dir, "credit_wise_model.pkl"),
            os.path.join(base_dir, "..", "supervise_ml", "linear_reg", "logistic_reg", "credit_wise_model.pkl"),
            "credit_wise_model.pkl"
        ]
        candidate_scaler_paths = [
            scaler_path,
            os.path.join(base_dir, "credit_wise_scaler.pkl"),
            os.path.join(base_dir, "..", "supervise_ml", "linear_reg", "logistic_reg", "credit_wise_scaler.pkl"),
            "credit_wise_scaler.pkl"
        ]

        self.model = None
        for p in candidate_model_paths:
            if p and os.path.exists(p):
                try:
                    self.model = joblib.load(p)
                    self.model_path = p
                    break
                except Exception:
                    continue

        if self.model is None:
            raise FileNotFoundError("Could not find or load credit_wise_model.pkl")

        self.scaler = None
        for p in candidate_scaler_paths:
            if p and os.path.exists(p):
                try:
                    self.scaler = joblib.load(p)
                    self.scaler_path = p
                    break
                except Exception:
                    continue

        if self.scaler is None:
            raise FileNotFoundError("Could not find or load credit_wise_scaler.pkl")

    def transform_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms raw user input dictionary into the 27 features formatted for the scaler.
        """
        # Numerical values with sane defaults
        app_income = float(data.get("Applicant_Income", 10000.0))
        coapp_income = float(data.get("Coapplicant_Income", 0.0))
        age = float(data.get("Age", 35.0))
        dependents = float(data.get("Dependents", 0.0))
        existing_loans = float(data.get("Existing_Loans", 1.0))
        savings = float(data.get("Savings", 8000.0))
        collateral = float(data.get("Collateral_Value", 20000.0))
        loan_amount = float(data.get("Loan_Amount", 20000.0))
        loan_term = float(data.get("Loan_Term", 36.0))
        credit_score = float(data.get("Credit_Score", 650.0))
        dti_ratio = float(data.get("DTI_Ratio", 0.35))

        # Categorical values
        education = str(data.get("Education_Level", "Graduate")).strip()
        employment = str(data.get("Employment_Status", "Salaried")).strip()
        marital = str(data.get("Marital_Status", "Single")).strip()
        loan_purpose = str(data.get("Loan_Purpose", "Personal")).strip()
        prop_area = str(data.get("Property_Area", "Urban")).strip()
        gender = str(data.get("Gender", "Male")).strip()
        employer_cat = str(data.get("Employer_Category", "Private")).strip()

        # LabelEncoder for Education_Level: 'Graduate' -> 0, 'Not Graduate' -> 1
        edu_code = 1 if "not" in education.lower() else 0

        # One-Hot Encodings (drop="first" logic from training notebook)
        emp_salaried = 1 if employment.lower() == "salaried" else 0
        emp_self = 1 if employment.lower() in ["self-employed", "self employed"] else 0
        emp_unemployed = 1 if employment.lower() == "unemployed" else 0

        marital_single = 1 if marital.lower() == "single" else 0

        purpose_car = 1 if loan_purpose.lower() == "car" else 0
        purpose_edu = 1 if loan_purpose.lower() == "education" else 0
        purpose_home = 1 if loan_purpose.lower() == "home" else 0
        purpose_personal = 1 if loan_purpose.lower() == "personal" else 0

        prop_semiurban = 1 if prop_area.lower() == "semiurban" else 0
        prop_urban = 1 if prop_area.lower() == "urban" else 0

        gender_male = 1 if gender.lower() == "male" else 0

        emp_cat_gov = 1 if employer_cat.lower() == "government" else 0
        emp_cat_mnc = 1 if employer_cat.lower() == "mnc" else 0
        emp_cat_pvt = 1 if employer_cat.lower() == "private" else 0
        emp_cat_unemployed = 1 if employer_cat.lower() == "unemployed" else 0

        # Non-linear squared features
        dti_ratio_sq = dti_ratio ** 2
        credit_score_sq = credit_score ** 2

        feature_dict = {
            "Applicant_Income": app_income,
            "Coapplicant_Income": coapp_income,
            "Age": age,
            "Dependents": dependents,
            "Existing_Loans": existing_loans,
            "Savings": savings,
            "Collateral_Value": collateral,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Education_Level": edu_code,
            "Employment_Status_Salaried": emp_salaried,
            "Employment_Status_Self-employed": emp_self,
            "Employment_Status_Unemployed": emp_unemployed,
            "Marital_Status_Single": marital_single,
            "Loan_Purpose_Car": purpose_car,
            "Loan_Purpose_Education": purpose_edu,
            "Loan_Purpose_Home": purpose_home,
            "Loan_Purpose_Personal": purpose_personal,
            "Property_Area_Semiurban": prop_semiurban,
            "Property_Area_Urban": prop_urban,
            "Gender_Male": gender_male,
            "Employer_Category_Government": emp_cat_gov,
            "Employer_Category_MNC": emp_cat_mnc,
            "Employer_Category_Private": emp_cat_pvt,
            "Employer_Category_Unemployed": emp_cat_unemployed,
            "DTI_Ratio_sq": dti_ratio_sq,
            "Credit_Score_sq": credit_score_sq
        }

        # Ensure exact column ordering as fitted in scaler
        feature_order = list(self.scaler.feature_names_in_) if hasattr(self.scaler, "feature_names_in_") else FEATURE_NAMES
        return pd.DataFrame([feature_dict], columns=feature_order)

    def predict(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs full inference pipeline and returns comprehensive prediction analytics.
        """
        features_df = self.transform_features(raw_data)
        scaled_features = self.scaler.transform(features_df)
        
        pred = int(self.model.predict(scaled_features)[0])
        probas = self.model.predict_proba(scaled_features)[0]
        
        prob_reject = float(probas[0])
        prob_approve = float(probas[1])

        approved = (pred == 1)
        decision = "APPROVED" if approved else "REJECTED"

        # Key analytical indicators
        credit_score = float(raw_data.get("Credit_Score", 650.0))
        dti = float(raw_data.get("DTI_Ratio", 0.35))
        loan_amt = float(raw_data.get("Loan_Amount", 20000.0))
        collateral = float(raw_data.get("Collateral_Value", 0.0))
        income = float(raw_data.get("Applicant_Income", 10000.0))
        coapp_income = float(raw_data.get("Coapplicant_Income", 0.0))
        savings = float(raw_data.get("Savings", 0.0))
        total_income = income + coapp_income

        coverage_ratio = round((collateral / loan_amt), 2) if loan_amt > 0 else 0.0
        income_to_loan = round((total_income / loan_amt), 2) if loan_amt > 0 else 0.0

        # Credit Score Tier
        if credit_score >= 750:
            credit_tier = "Excellent"
        elif credit_score >= 670:
            credit_tier = "Good"
        elif credit_score >= 580:
            credit_tier = "Fair"
        else:
            credit_tier = "Poor"

        # DTI Tier
        if dti <= 0.30:
            dti_status = "Healthy (<30%)"
        elif dti <= 0.45:
            dti_status = "Moderate (30-45%)"
        else:
            dti_status = "High Debt Burden (>45%)"

        # Risk Tier
        if prob_approve >= 0.70:
            risk_tier = "Low Risk"
        elif prob_approve >= 0.40:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "High Risk"

        # Key Drivers Analysis
        strengths: List[str] = []
        risks: List[str] = []
        recommendations: List[str] = []

        if credit_score >= 680:
            strengths.append(f"Strong credit score ({int(credit_score)}) indicates high repayment reliability.")
        else:
            risks.append(f"Credit score ({int(credit_score)}) is below prime lending threshold.")
            recommendations.append("Work on improving credit score by paying off delinquent accounts and reducing credit card balances.")

        if dti <= 0.35:
            strengths.append(f"Low Debt-to-Income ratio ({dti:.2f}) indicates manageable monthly obligations.")
        else:
            risks.append(f"High Debt-to-Income ratio ({dti:.2f}) limits available cash flow for debt service.")
            recommendations.append("Consider consolidating or paying down existing credit lines to bring DTI ratio under 0.35.")

        if coverage_ratio >= 1.0:
            strengths.append(f"Solid collateral coverage ({coverage_ratio}x requested loan amount).")
        elif coverage_ratio < 0.5:
            risks.append(f"Low collateral coverage ({coverage_ratio}x loan amount).")
            recommendations.append("Pledge additional collateral or reduce loan amount to improve loan security.")

        if savings >= (loan_amt * 0.25):
            strengths.append(f"Healthy emergency savings reserve (${savings:,.0f}).")
        else:
            risks.append(f"Limited savings buffer (${savings:,.0f}) relative to requested loan.")

        if not strengths:
            strengths.append("Applicant has active income source.")
        if not risks and not approved:
            risks.append("Overall risk score based on historical default patterns.")
        if not recommendations:
            recommendations.append("Maintain timely payments and avoid taking on new short-term debt.")

        return {
            "loan_approved": approved,
            "decision": decision,
            "approval_probability": round(prob_approve * 100, 2),
            "rejection_probability": round(prob_reject * 100, 2),
            "risk_tier": risk_tier,
            "metrics": {
                "credit_score": int(credit_score),
                "credit_score_tier": credit_tier,
                "dti_ratio": round(dti, 3),
                "dti_status": dti_status,
                "collateral_coverage_ratio": coverage_ratio,
                "income_to_loan_ratio": income_to_loan,
                "total_monthly_income": total_income
            },
            "insights": {
                "strengths": strengths,
                "risks": risks,
                "recommendations": recommendations
            }
        }


def get_sample_scenarios() -> Dict[str, Dict[str, Any]]:
    return {
        "prime_approved": {
            "name": "Prime Approved Applicant",
            "description": "High credit score, low DTI, salaried MNC professional with substantial collateral.",
            "data": {
                "Applicant_Income": 16000.0,
                "Coapplicant_Income": 6000.0,
                "Age": 42.0,
                "Dependents": 1.0,
                "Existing_Loans": 1.0,
                "Savings": 18500.0,
                "Collateral_Value": 45000.0,
                "Loan_Amount": 15000.0,
                "Loan_Term": 48.0,
                "Credit_Score": 765.0,
                "DTI_Ratio": 0.22,
                "Education_Level": "Graduate",
                "Employment_Status": "Salaried",
                "Marital_Status": "Married",
                "Loan_Purpose": "Home",
                "Property_Area": "Urban",
                "Gender": "Male",
                "Employer_Category": "MNC"
            }
        },
        "moderate_borderline": {
            "name": "Moderate Risk Applicant",
            "description": "Average income, moderate credit score and DTI, self-employed.",
            "data": {
                "Applicant_Income": 9500.0,
                "Coapplicant_Income": 2500.0,
                "Age": 33.0,
                "Dependents": 2.0,
                "Existing_Loans": 2.0,
                "Savings": 8000.0,
                "Collateral_Value": 18000.0,
                "Loan_Amount": 20000.0,
                "Loan_Term": 36.0,
                "Credit_Score": 665.0,
                "DTI_Ratio": 0.38,
                "Education_Level": "Graduate",
                "Employment_Status": "Self-employed",
                "Marital_Status": "Single",
                "Loan_Purpose": "Car",
                "Property_Area": "Semiurban",
                "Gender": "Female",
                "Employer_Category": "Private"
            }
        },
        "high_risk_rejected": {
            "name": "High Risk Applicant",
            "description": "Subprime credit score, high DTI ratio, 4 existing loans, low collateral.",
            "data": {
                "Applicant_Income": 4500.0,
                "Coapplicant_Income": 0.0,
                "Age": 26.0,
                "Dependents": 3.0,
                "Existing_Loans": 4.0,
                "Savings": 1200.0,
                "Collateral_Value": 5000.0,
                "Loan_Amount": 32000.0,
                "Loan_Term": 72.0,
                "Credit_Score": 560.0,
                "DTI_Ratio": 0.58,
                "Education_Level": "Not Graduate",
                "Employment_Status": "Unemployed",
                "Marital_Status": "Single",
                "Loan_Purpose": "Personal",
                "Property_Area": "Rural",
                "Gender": "Male",
                "Employer_Category": "Unemployed"
            }
        }
    }

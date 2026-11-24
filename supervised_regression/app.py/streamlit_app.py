import io
import streamlit as st
import pandas as pd
from predictor import CreditWisePredictor, get_sample_scenarios

# Page Configuration
st.set_page_config(
    page_title="CreditWise AI | Smart Loan Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.25rem;
        border-radius: 1rem;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .status-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
        border: 1px solid #10b981;
        padding: 1rem;
        border-radius: 0.75rem;
        color: #34d399;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
    }
    .status-rejected {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border: 1px solid #ef4444;
        padding: 1rem;
        border-radius: 0.75rem;
        color: #f87171;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_color_fallback=True)

# Cache predictor singleton
@st.cache_resource
def load_predictor():
    return CreditWisePredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load model or scaler: {e}")
    st.stop()

# Title
st.markdown('<div class="main-title">CreditWise AI — Loan Approval Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Evaluate applicant creditworthiness using trained Gaussian Naive Bayes and calibrated 27-feature Standard Scaler.</div>', unsafe_allow_html=True)

# Tabs
tab_single, tab_batch = st.tabs(["📝 Single Applicant Evaluation", "📂 Batch CSV Prediction"])

with tab_single:
    scenarios = get_sample_scenarios()

    # Preset selection
    st.markdown("##### ⚡ Quick Scenarios")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    if "form_data" not in st.session_state:
        st.session_state.form_data = scenarios["prime_approved"]["data"].copy()

    with preset_col1:
        if st.button("🌟 Load Prime Approved Applicant", use_container_width=True):
            st.session_state.form_data = scenarios["prime_approved"]["data"].copy()
            st.rerun()

    with preset_col2:
        if st.button("⚠️ Load Moderate Risk Applicant", use_container_width=True):
            st.session_state.form_data = scenarios["moderate_borderline"]["data"].copy()
            st.rerun()

    with preset_col3:
        if st.button("❌ Load High Risk (Reject) Applicant", use_container_width=True):
            st.session_state.form_data = scenarios["high_risk_rejected"]["data"].copy()
            st.rerun()

    st.divider()

    cur = st.session_state.form_data

    # Input Form Layout in 3 Columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=int(cur.get("Age", 38)))
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if cur.get("Gender") == "Male" else 1)
        marital = st.selectbox("Marital Status", ["Married", "Single"], index=0 if cur.get("Marital_Status") == "Married" else 1)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=int(cur.get("Dependents", 1)))
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], index=0 if cur.get("Education_Level") == "Graduate" else 1)
        prop_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban", "Semiurban", "Rural"].index(cur.get("Property_Area", "Urban")))

    with col2:
        st.subheader("💼 Employment & Income")
        emp_options = ["Salaried", "Self-employed", "Contract", "Unemployed"]
        emp_status = st.selectbox("Employment Status", emp_options, index=emp_options.index(cur.get("Employment_Status", "Salaried")) if cur.get("Employment_Status") in emp_options else 0)
        
        cat_options = ["Private", "Government", "MNC", "Business", "Unemployed"]
        emp_cat = st.selectbox("Employer Category", cat_options, index=cat_options.index(cur.get("Employer_Category", "Private")) if cur.get("Employer_Category") in cat_options else 0)

        app_income = st.number_input("Applicant Income ($/mo)", min_value=0.0, step=500.0, value=float(cur.get("Applicant_Income", 14500.0)))
        coapp_income = st.number_input("Coapplicant Income ($/mo)", min_value=0.0, step=500.0, value=float(cur.get("Coapplicant_Income", 4000.0)))

    with col3:
        st.subheader("💳 Credit & Loan Request")
        credit_score = st.slider("Credit Score (FICO/CIBIL)", min_value=300, max_value=850, value=int(cur.get("Credit_Score", 735)))
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, step=0.01, value=float(cur.get("DTI_Ratio", 0.25)))
        existing_loans = st.number_input("Existing Active Loans", min_value=0, max_value=10, value=int(cur.get("Existing_Loans", 1)))
        savings = st.number_input("Liquid Savings ($)", min_value=0.0, step=1000.0, value=float(cur.get("Savings", 16000.0)))
        collateral = st.number_input("Collateral Value ($)", min_value=0.0, step=1000.0, value=float(cur.get("Collateral_Value", 38000.0)))
        
        purpose_options = ["Home", "Personal", "Car", "Business", "Education"]
        loan_purpose = st.selectbox("Loan Purpose", purpose_options, index=purpose_options.index(cur.get("Loan_Purpose", "Home")) if cur.get("Loan_Purpose") in purpose_options else 0)
        
        loan_amt = st.number_input("Requested Loan Amount ($)", min_value=1000.0, step=1000.0, value=float(cur.get("Loan_Amount", 20000.0)))
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84], index=3)

    st.write("")
    eval_btn = st.button("🚀 Evaluate Loan Application", type="primary", use_container_width=True)

    if eval_btn:
        payload = {
            "Applicant_Income": app_income,
            "Coapplicant_Income": coapp_income,
            "Age": age,
            "Dependents": dependents,
            "Existing_Loans": existing_loans,
            "Savings": savings,
            "Collateral_Value": collateral,
            "Loan_Amount": loan_amt,
            "Loan_Term": loan_term,
            "Credit_Score": credit_score,
            "DTI_Ratio": dti_ratio,
            "Education_Level": education,
            "Employment_Status": emp_status,
            "Marital_Status": marital,
            "Loan_Purpose": loan_purpose,
            "Property_Area": prop_area,
            "Gender": gender,
            "Employer_Category": emp_cat
        }

        with st.spinner("Processing underwriting models..."):
            res = predictor.predict(payload)

        st.divider()
        st.subheader("📊 Underwriting Decision & Risk Report")

        # Top Results Banner
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            if res["loan_approved"]:
                st.markdown('<div class="status-approved">✅ APPROVED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-rejected">❌ REJECTED</div>', unsafe_allow_html=True)
        with m_col2:
            st.metric("Approval Probability", f"{res['approval_probability']}%")
            st.progress(res["approval_probability"] / 100.0)
        with m_col3:
            st.metric("Risk Classification", res["risk_tier"])
        with m_col4:
            st.metric("Credit Bureau Tier", f"{res['metrics']['credit_score_tier']} ({res['metrics']['credit_score']})")

        # Additional financial metrics
        st.markdown("##### 📈 Financial Health Indicators")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.info(f"**DTI Health:** {res['metrics']['dti_ratio']} — {res['metrics']['dti_status']}")
        with f_col2:
            st.info(f"**Collateral Coverage:** {res['metrics']['collateral_coverage_ratio']}x Loan Principal")
        with f_col3:
            st.info(f"**Total Household Income:** ${res['metrics']['total_monthly_income']:,.0f} / month")

        # Insights
        i_col1, i_col2, i_col3 = st.columns(3)
        with i_col1:
            st.success("**Key Strengths:**")
            for s in res["insights"]["strengths"]:
                st.write(f"• {s}")
        with i_col2:
            st.warning("**Identified Risk Factors:**")
            if res["insights"]["risks"]:
                for r in res["insights"]["risks"]:
                    st.write(f"• {r}")
            else:
                st.write("• No critical risk flags detected.")
        with i_col3:
            st.info("**Advisor Recommendations:**")
            for rec in res["insights"]["recommendations"]:
                st.write(f"• {rec}")


with tab_batch:
    st.subheader("📂 Batch Loan Scoring")
    st.markdown("Upload a CSV file containing applicant records. The system will compute scaled features, predictions, and approval probabilities for each row.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded **{len(batch_df)}** records:")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("⚡ Score All Applicants"):
                with st.spinner("Scoring dataset..."):
                    results_list = []
                    for _, row in batch_df.iterrows():
                        try:
                            pred_res = predictor.predict(row.to_dict())
                            results_list.append({
                                "Loan_Approved": "Yes" if pred_res["loan_approved"] else "No",
                                "Decision": pred_res["decision"],
                                "Approval_Probability_%": pred_res["approval_probability"],
                                "Risk_Tier": pred_res["risk_tier"]
                            })
                        except Exception:
                            results_list.append({
                                "Loan_Approved": "Error",
                                "Decision": "ERROR",
                                "Approval_Probability_%": 0.0,
                                "Risk_Tier": "Unknown"
                            })

                    scored_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(results_list)], axis=1)

                    st.success("Batch scoring completed!")
                    st.dataframe(scored_df.head(20), use_container_width=True)

                    # Download button
                    csv_buf = io.StringIO()
                    scored_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="📥 Download Scored CSV",
                        data=csv_buf.getvalue(),
                        file_name="creditwise_scored_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")

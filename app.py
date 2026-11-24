import os
import sys

# Ensure this file's directory is the active directory and in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
try:
    os.chdir(BASE_DIR)
except Exception:
    pass

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

from predictor import CreditWisePredictor, get_sample_scenarios

# Initialize FastAPI App
app = FastAPI(
    title="CreditWise Loan Approval API & UI",
    description="Intelligent Credit Eligibility & Risk Prediction using Machine Learning",
    version="1.0.0"
)

# Enable CORS for external frontends or cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
predictor = CreditWisePredictor()


# Pydantic schema for loan request validation
class LoanApplicationRequest(BaseModel):
    Applicant_Income: float = Field(12000.0, ge=0, description="Primary applicant monthly income in USD")
    Coapplicant_Income: float = Field(3500.0, ge=0, description="Co-applicant monthly income in USD")
    Age: float = Field(38.0, ge=18, le=100, description="Applicant age in years")
    Dependents: float = Field(1.0, ge=0, le=15, description="Number of financial dependents")
    Existing_Loans: float = Field(1.0, ge=0, le=20, description="Number of currently active loans")
    Savings: float = Field(15000.0, ge=0, description="Total liquid savings/checking balance")
    Collateral_Value: float = Field(35000.0, ge=0, description="Appraised asset/collateral value")
    Loan_Amount: float = Field(22000.0, gt=0, description="Requested loan principal amount")
    Loan_Term: float = Field(48.0, ge=6, le=360, description="Loan term in months")
    Credit_Score: float = Field(710.0, ge=300, le=850, description="Credit Bureau Score (FICO/CIBIL)")
    DTI_Ratio: float = Field(0.28, ge=0.0, le=1.5, description="Debt-to-Income ratio (0.0 to 1.0+)")
    Education_Level: str = Field("Graduate", description="'Graduate' or 'Not Graduate'")
    Employment_Status: str = Field("Salaried", description="'Salaried', 'Self-employed', 'Contract', 'Unemployed'")
    Marital_Status: str = Field("Married", description="'Married' or 'Single'")
    Loan_Purpose: str = Field("Home", description="'Personal', 'Car', 'Business', 'Home', 'Education'")
    Property_Area: str = Field("Urban", description="'Urban', 'Semiurban', 'Rural'")
    Gender: str = Field("Male", description="'Male' or 'Female'")
    Employer_Category: str = Field("Private", description="'Private', 'Government', 'MNC', 'Business', 'Unemployed'")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


@app.get("/api/health")
def health_check():
    """Healthcheck endpoint returning model status and active features."""
    return {
        "status": "healthy",
        "model_type": str(type(predictor.model).__name__),
        "scaler_type": str(type(predictor.scaler).__name__),
        "feature_count": len(predictor.scaler.feature_names_in_) if hasattr(predictor.scaler, "feature_names_in_") else 27,
        "classes": [int(c) for c in predictor.model.classes_] if hasattr(predictor.model, "classes_") else [0, 1]
    }


@app.get("/api/scenarios")
def get_scenarios():
    """Returns sample pre-configured applicant profiles for testing."""
    return get_sample_scenarios()


@app.post("/api/predict")
def predict_loan(application: LoanApplicationRequest):
    """
    Main machine learning prediction endpoint.
    Processes applicant data, scales features, and returns loan approval decision with analytics.
    """
    try:
        raw_dict = application.model_dump()
        result = predictor.predict(raw_dict)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the complete responsive CreditWise Web Application."""
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CreditWise AI | Smart Loan Approval & Risk Assessment</title>
  <!-- Tailwind CSS & Google Fonts -->
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/lucide@latest"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['"Plus Jakarta Sans"', 'sans-serif'],
          },
          colors: {
            brand: {
              50: '#eef2ff',
              100: '#e0e7ff',
              500: '#6366f1',
              600: '#4f46e5',
              700: '#4338ca',
              900: '#1e1b4b'
            }
          }
        }
      }
    }
  </script>
  <style>
    body {
      background: radial-gradient(circle at 50% 0%, #1e1e2f 0%, #0d0f17 100%);
      color: #e2e8f0;
      min-height: 100vh;
    }
    .glass-card {
      background: rgba(22, 27, 44, 0.7);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-input {
      background: rgba(15, 20, 35, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #f8fafc;
      transition: all 0.2s ease;
    }
    .glass-input:focus {
      border-color: #6366f1;
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
      outline: none;
    }
    .glow-approved {
      box-shadow: 0 0 35px -5px rgba(16, 185, 129, 0.35);
    }
    .glow-rejected {
      box-shadow: 0 0 35px -5px rgba(239, 68, 68, 0.35);
    }
  </style>
</head>
<body class="font-sans antialiased selection:bg-indigo-500 selection:text-white pb-16">

  <!-- Navigation Bar -->
  <nav class="border-b border-white/10 glass-card sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
          <i data-lucide="shield-check" class="w-6 h-6 text-white"></i>
        </div>
        <div>
          <span class="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-indigo-100 to-indigo-300 bg-clip-text text-transparent">CreditWise AI</span>
          <span class="hidden sm:inline-block text-xs px-2 py-0.5 ml-2 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium">ML v1.0</span>
        </div>
      </div>
      <div class="flex items-center space-x-4">
        <a href="/docs" target="_blank" class="text-xs font-semibold px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-all flex items-center space-x-1.5 border border-white/10">
          <i data-lucide="code" class="w-3.5 h-3.5"></i>
          <span>Swagger API Docs</span>
        </a>
        <div class="flex items-center space-x-2 text-xs font-medium text-emerald-400 bg-emerald-500/10 px-3 py-1.5 rounded-full border border-emerald-500/20">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>Engine Online</span>
        </div>
      </div>
    </div>
  </nav>

  <!-- Hero Header -->
  <header class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 pb-6 text-center sm:text-left">
    <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
      <div>
        <h1 class="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
          Intelligent Loan Eligibility Assessment
        </h1>
        <p class="text-slate-400 mt-2 text-sm sm:text-base max-w-2xl">
          Powered by a Gaussian Naive Bayes classifier & calibrated standard scaler. Evaluate applicant creditworthiness, risk probabilities, and automated decision rationales.
        </p>
      </div>
      <!-- Preset Buttons -->
      <div class="flex flex-wrap items-center gap-2 self-start sm:self-auto">
        <span class="text-xs text-slate-400 mr-1 font-medium">Quick Presets:</span>
        <button onclick="loadScenario('prime_approved')" class="text-xs font-medium px-3 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 transition flex items-center space-x-1">
          <i data-lucide="user-check" class="w-3.5 h-3.5"></i>
          <span>Prime Approved</span>
        </button>
        <button onclick="loadScenario('moderate_borderline')" class="text-xs font-medium px-3 py-1.5 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/30 transition flex items-center space-x-1">
          <i data-lucide="alert-circle" class="w-3.5 h-3.5"></i>
          <span>Moderate Risk</span>
        </button>
        <button onclick="loadScenario('high_risk_rejected')" class="text-xs font-medium px-3 py-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30 transition flex items-center space-x-1">
          <i data-lucide="user-x" class="w-3.5 h-3.5"></i>
          <span>High Risk (Reject)</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 lg:grid-cols-12 gap-8 mt-2">
    
    <!-- Application Input Form (7 Cols) -->
    <div class="lg:col-span-7 space-y-6">
      <form id="loanForm" onsubmit="handleFormSubmit(event)" class="space-y-6">
        
        <!-- Section 1: Demographics & Personal -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <div class="flex items-center space-x-2 text-indigo-400 mb-4 font-semibold text-sm">
            <i data-lucide="user" class="w-4 h-4"></i>
            <span>Applicant Demographics</span>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Age</label>
              <input type="number" id="Age" value="38" min="18" max="99" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Gender</label>
              <select id="Gender" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Male" selected>Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Marital Status</label>
              <select id="Marital_Status" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Married" selected>Married</option>
                <option value="Single">Single</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Dependents</label>
              <input type="number" id="Dependents" value="1" min="0" max="10" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Education Level</label>
              <select id="Education_Level" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Graduate" selected>Graduate</option>
                <option value="Not Graduate">Not Graduate</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Property Area</label>
              <select id="Property_Area" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Urban" selected>Urban</option>
                <option value="Semiurban">Semiurban</option>
                <option value="Rural">Rural</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Section 2: Employment & Income -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <div class="flex items-center space-x-2 text-indigo-400 mb-4 font-semibold text-sm">
            <i data-lucide="briefcase" class="w-4 h-4"></i>
            <span>Employment & Monthly Incomes</span>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Employment Status</label>
              <select id="Employment_Status" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Salaried" selected>Salaried</option>
                <option value="Self-employed">Self-employed</option>
                <option value="Contract">Contract</option>
                <option value="Unemployed">Unemployed</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Employer Category</label>
              <select id="Employer_Category" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Private" selected>Private</option>
                <option value="Government">Government</option>
                <option value="MNC">MNC</option>
                <option value="Business">Business</option>
                <option value="Unemployed">Unemployed</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Applicant Income ($/mo)</label>
              <input type="number" id="Applicant_Income" value="14500" min="0" step="100" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Coapplicant Income ($/mo)</label>
              <input type="number" id="Coapplicant_Income" value="4000" min="0" step="100" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
          </div>
        </div>

        <!-- Section 3: Financial Health & Loan Request -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <div class="flex items-center space-x-2 text-indigo-400 mb-4 font-semibold text-sm">
            <i data-lucide="credit-card" class="w-4 h-4"></i>
            <span>Credit History & Loan Details</span>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Credit Score (300-850)</label>
              <input type="number" id="Credit_Score" value="735" min="300" max="850" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">DTI Ratio (e.g. 0.25)</label>
              <input type="number" id="DTI_Ratio" value="0.25" min="0.0" max="1.0" step="0.01" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Existing Loans</label>
              <input type="number" id="Existing_Loans" value="1" min="0" max="10" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Liquid Savings ($)</label>
              <input type="number" id="Savings" value="16000" min="0" step="100" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Collateral Value ($)</label>
              <input type="number" id="Collateral_Value" value="38000" min="0" step="100" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Loan Purpose</label>
              <select id="Loan_Purpose" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Home" selected>Home</option>
                <option value="Personal">Personal</option>
                <option value="Car">Car</option>
                <option value="Business">Business</option>
                <option value="Education">Education</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Requested Loan ($)</label>
              <input type="number" id="Loan_Amount" value="20000" min="500" step="500" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Loan Term (Months)</label>
              <select id="Loan_Term" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="12">12 Months (1 yr)</option>
                <option value="24">24 Months (2 yrs)</option>
                <option value="36">36 Months (3 yrs)</option>
                <option value="48" selected>48 Months (4 yrs)</option>
                <option value="60">60 Months (5 yrs)</option>
                <option value="72">72 Months (6 yrs)</option>
                <option value="84">84 Months (7 yrs)</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Submit Button -->
        <button type="submit" id="submitBtn" class="w-full py-4 px-6 rounded-2xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-bold text-base shadow-lg shadow-indigo-600/30 transition-all flex items-center justify-center space-x-2">
          <i data-lucide="sparkles" class="w-5 h-5"></i>
          <span>Evaluate Loan Eligibility</span>
        </button>
      </form>
    </div>

    <!-- Prediction & Analysis Results Panel (5 Cols) -->
    <div class="lg:col-span-5 space-y-6">
      
      <!-- Primary Decision Card -->
      <div id="resultCard" class="glass-card rounded-2xl p-6 shadow-xl transition-all duration-300">
        <div class="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Automated Assessment</div>
        
        <!-- Decision Header -->
        <div class="flex items-center justify-between pb-6 border-b border-white/10">
          <div>
            <div id="decisionBadge" class="text-3xl font-extrabold tracking-tight text-white flex items-center space-x-2">
              <span>Ready for Review</span>
            </div>
            <div id="riskTierText" class="text-xs text-slate-400 mt-1 font-medium">Fill application form and evaluate</div>
          </div>
          <div id="decisionIcon" class="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center border border-white/10">
            <i data-lucide="calculator" class="w-7 h-7 text-indigo-400"></i>
          </div>
        </div>

        <!-- Probability Meter -->
        <div class="mt-6">
          <div class="flex justify-between items-center text-xs mb-1.5">
            <span class="text-slate-300 font-medium">Approval Probability</span>
            <span id="probPercentage" class="text-indigo-400 font-bold">--%</span>
          </div>
          <div class="w-full h-3 rounded-full bg-slate-800 overflow-hidden p-0.5 border border-white/5">
            <div id="probBar" class="h-full rounded-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-700" style="width: 0%"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>High Risk (0%)</span>
            <span>Threshold (50%)</span>
            <span>Prime (100%)</span>
          </div>
        </div>

        <!-- Key Financial Metrics Grid -->
        <div class="grid grid-cols-2 gap-3 mt-6">
          <div class="p-3 rounded-xl bg-white/5 border border-white/5">
            <div class="text-[11px] text-slate-400">Credit Score Tier</div>
            <div id="metricCreditTier" class="text-sm font-bold text-slate-200 mt-0.5">--</div>
          </div>
          <div class="p-3 rounded-xl bg-white/5 border border-white/5">
            <div class="text-[11px] text-slate-400">DTI Health</div>
            <div id="metricDti" class="text-sm font-bold text-slate-200 mt-0.5">--</div>
          </div>
          <div class="p-3 rounded-xl bg-white/5 border border-white/5">
            <div class="text-[11px] text-slate-400">Collateral Coverage</div>
            <div id="metricCoverage" class="text-sm font-bold text-slate-200 mt-0.5">--</div>
          </div>
          <div class="p-3 rounded-xl bg-white/5 border border-white/5">
            <div class="text-[11px] text-slate-400">Total Monthly Income</div>
            <div id="metricIncome" class="text-sm font-bold text-slate-200 mt-0.5">--</div>
          </div>
        </div>

        <!-- Insights Accordion / Lists -->
        <div class="mt-6 space-y-4">
          <!-- Strengths -->
          <div>
            <div class="text-xs font-semibold text-emerald-400 flex items-center space-x-1.5 mb-2">
              <i data-lucide="check-circle" class="w-3.5 h-3.5"></i>
              <span>Key Strengths</span>
            </div>
            <ul id="strengthsList" class="text-xs text-slate-300 space-y-1.5 list-disc pl-4">
              <li class="text-slate-500 italic list-none">Submit applicant profile to view strengths.</li>
            </ul>
          </div>

          <!-- Risk Factors -->
          <div>
            <div class="text-xs font-semibold text-rose-400 flex items-center space-x-1.5 mb-2">
              <i data-lucide="alert-triangle" class="w-3.5 h-3.5"></i>
              <span>Identified Risk Factors</span>
            </div>
            <ul id="risksList" class="text-xs text-slate-300 space-y-1.5 list-disc pl-4">
              <li class="text-slate-500 italic list-none">Submit applicant profile to view risk flags.</li>
            </ul>
          </div>

          <!-- Recommendations -->
          <div>
            <div class="text-xs font-semibold text-indigo-400 flex items-center space-x-1.5 mb-2">
              <i data-lucide="lightbulb" class="w-3.5 h-3.5"></i>
              <span>Advisor Recommendations</span>
            </div>
            <ul id="recsList" class="text-xs text-slate-300 space-y-1.5 list-disc pl-4">
              <li class="text-slate-500 italic list-none">Submit applicant profile to view recommendations.</li>
            </ul>
          </div>
        </div>

      </div>

      <!-- Architecture Card -->
      <div class="glass-card rounded-2xl p-5 border border-white/5 text-xs text-slate-400">
        <div class="font-semibold text-slate-200 flex items-center space-x-1.5 mb-2">
          <i data-lucide="cpu" class="w-4 h-4 text-indigo-400"></i>
          <span>Model Architecture Specs</span>
        </div>
        <p class="leading-relaxed">
          Gaussian Naive Bayes classifier trained on historical credit records with 27 standard-scaled features including non-linear interaction terms (<code class="text-indigo-300">Credit_Score²</code>, <code class="text-indigo-300">DTI_Ratio²</code>).
        </p>
      </div>

    </div>
  </main>

  <script>
    lucide.createIcons();

    // Sample Scenarios Cache
    let scenarios = null;

    async function fetchScenarios() {
      try {
        const res = await fetch('/api/scenarios');
        scenarios = await res.json();
      } catch (err) {
        console.error('Error loading scenarios:', err);
      }
    }
    fetchScenarios();

    function loadScenario(key) {
      if (!scenarios || !scenarios[key]) return;
      const data = scenarios[key].data;
      for (const [k, v] of Object.entries(data)) {
        const el = document.getElementById(k);
        if (el) {
          el.value = v;
        }
      }
      handleFormSubmit(new Event('submit'));
    }

    async function handleFormSubmit(e) {
      if (e) e.preventDefault();
      
      const btn = document.getElementById('submitBtn');
      const originalText = btn.innerHTML;
      btn.innerHTML = '<span class="animate-spin inline-block mr-2">&#8635;</span> Evaluating Model...';
      btn.disabled = true;

      const payload = {
        Applicant_Income: parseFloat(document.getElementById('Applicant_Income').value),
        Coapplicant_Income: parseFloat(document.getElementById('Coapplicant_Income').value),
        Age: parseFloat(document.getElementById('Age').value),
        Dependents: parseFloat(document.getElementById('Dependents').value),
        Existing_Loans: parseFloat(document.getElementById('Existing_Loans').value),
        Savings: parseFloat(document.getElementById('Savings').value),
        Collateral_Value: parseFloat(document.getElementById('Collateral_Value').value),
        Loan_Amount: parseFloat(document.getElementById('Loan_Amount').value),
        Loan_Term: parseFloat(document.getElementById('Loan_Term').value),
        Credit_Score: parseFloat(document.getElementById('Credit_Score').value),
        DTI_Ratio: parseFloat(document.getElementById('DTI_Ratio').value),
        Education_Level: document.getElementById('Education_Level').value,
        Employment_Status: document.getElementById('Employment_Status').value,
        Marital_Status: document.getElementById('Marital_Status').value,
        Loan_Purpose: document.getElementById('Loan_Purpose').value,
        Property_Area: document.getElementById('Property_Area').value,
        Gender: document.getElementById('Gender').value,
        Employer_Category: document.getElementById('Employer_Category').value
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const json = await res.json();
        if (json.success) {
          renderResult(json.data);
        } else {
          alert('Prediction failed: ' + (json.detail || 'Unknown error'));
        }
      } catch (err) {
        alert('Server connection error: ' + err.message);
      } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
        lucide.createIcons();
      }
    }

    function renderResult(data) {
      const card = document.getElementById('resultCard');
      const badge = document.getElementById('decisionBadge');
      const riskText = document.getElementById('riskTierText');
      const icon = document.getElementById('decisionIcon');
      const probText = document.getElementById('probPercentage');
      const probBar = document.getElementById('probBar');

      card.classList.remove('glow-approved', 'glow-rejected');

      if (data.loan_approved) {
        card.classList.add('glow-approved');
        badge.innerHTML = `<span class="text-emerald-400">APPROVED</span>`;
        icon.innerHTML = `<i data-lucide="check-check" class="w-8 h-8 text-emerald-400"></i>`;
        icon.className = 'w-14 h-14 rounded-2xl bg-emerald-500/20 flex items-center justify-center border border-emerald-500/40';
        probBar.className = 'h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-300 transition-all duration-700';
      } else {
        card.classList.add('glow-rejected');
        badge.innerHTML = `<span class="text-rose-400">REJECTED</span>`;
        icon.innerHTML = `<i data-lucide="x-circle" class="w-8 h-8 text-rose-400"></i>`;
        icon.className = 'w-14 h-14 rounded-2xl bg-rose-500/20 flex items-center justify-center border border-rose-500/40';
        probBar.className = 'h-full rounded-full bg-gradient-to-r from-rose-500 to-amber-500 transition-all duration-700';
      }

      riskText.textContent = `Confidence: ${data.approval_probability}% Approval • ${data.risk_tier}`;
      probText.textContent = `${data.approval_probability}%`;
      probBar.style.width = `${Math.min(Math.max(data.approval_probability, 3), 100)}%`;

      // Metrics
      document.getElementById('metricCreditTier').textContent = `${data.metrics.credit_score} (${data.metrics.credit_score_tier})`;
      document.getElementById('metricDti').textContent = `${data.metrics.dti_ratio} (${data.metrics.dti_status})`;
      document.getElementById('metricCoverage').textContent = `${data.metrics.collateral_coverage_ratio}x Loan Value`;
      document.getElementById('metricIncome').textContent = `$${data.metrics.total_monthly_income.toLocaleString()} / mo`;

      // Lists
      populateList('strengthsList', data.insights.strengths, 'No major strengths noted.');
      populateList('risksList', data.insights.risks, 'No high-risk flags identified.');
      populateList('recsList', data.insights.recommendations, 'Standard underwriting terms apply.');

      lucide.createIcons();
    }

    function populateList(elementId, items, emptyMessage) {
      const el = document.getElementById(elementId);
      el.innerHTML = '';
      if (!items || items.length === 0) {
        el.innerHTML = `<li class="text-slate-500 italic list-none">${emptyMessage}</li>`;
        return;
      }
      items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        el.appendChild(li);
      });
    }

    // Auto-run prime scenario on initial load
    window.addEventListener('DOMContentLoaded', () => {
      setTimeout(() => {
        loadScenario('prime_approved');
      }, 600);
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  🚀 CreditWise AI — Loan Approval Engine is LIVE!")
    print("  👉 Web Dashboard: http://127.0.0.1:8000")
    print("  👉 Swagger API:   http://127.0.0.1:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Initialize FastAPI App
app = FastAPI(
    title="CreditWise Loan Approval API & UI",
    description="Intelligent Credit Eligibility & Risk Prediction using Machine Learning",
    version="1.0.0"
)

# Enable CORS for external frontends or cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
predictor = CreditWisePredictor()


# Pydantic schema for loan request validation
class LoanApplicationRequest(BaseModel):
    Applicant_Income: float = Field(12000.0, ge=0, description="Primary applicant monthly income in USD")
    Coapplicant_Income: float = Field(3500.0, ge=0, description="Co-applicant monthly income in USD")
    Age: float = Field(38.0, ge=18, le=100, description="Applicant age in years")
    Dependents: float = Field(1.0, ge=0, le=15, description="Number of financial dependents")
    Existing_Loans: float = Field(1.0, ge=0, le=20, description="Number of currently active loans")
    Savings: float = Field(15000.0, ge=0, description="Total liquid savings/checking balance")
    Collateral_Value: float = Field(35000.0, ge=0, description="Appraised asset/collateral value")
    Loan_Amount: float = Field(22000.0, gt=0, description="Requested loan principal amount")
    Loan_Term: float = Field(48.0, ge=6, le=360, description="Loan term in months")
    Credit_Score: float = Field(710.0, ge=300, le=850, description="Credit Bureau Score (FICO/CIBIL)")
    DTI_Ratio: float = Field(0.28, ge=0.0, le=1.5, description="Debt-to-Income ratio (0.0 to 1.0+)")
    Education_Level: str = Field("Graduate", description="'Graduate' or 'Not Graduate'")
    Employment_Status: str = Field("Salaried", description="'Salaried', 'Self-employed', 'Contract', 'Unemployed'")
    Marital_Status: str = Field("Married", description="'Married' or 'Single'")
    Loan_Purpose: str = Field("Home", description="'Personal', 'Car', 'Business', 'Home', 'Education'")
    Property_Area: str = Field("Urban", description="'Urban', 'Semiurban', 'Rural'")
    Gender: str = Field("Male", description="'Male' or 'Female'")
    Employer_Category: str = Field("Private", description="'Private', 'Government', 'MNC', 'Business', 'Unemployed'")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


@app.get("/api/health")
def health_check():
    """Healthcheck endpoint returning model status and active features."""
    return {
        "status": "healthy",
        "model_type": str(type(predictor.model).__name__),
        "scaler_type": str(type(predictor.scaler).__name__),
        "feature_count": len(predictor.scaler.feature_names_in_) if hasattr(predictor.scaler, "feature_names_in_") else 27,
        "classes": [int(c) for c in predictor.model.classes_] if hasattr(predictor.model, "classes_") else [0, 1]
    }


@app.get("/api/scenarios")
def get_scenarios():
    """Returns sample pre-configured applicant profiles for testing."""
    return get_sample_scenarios()


@app.post("/api/predict")
def predict_loan(application: LoanApplicationRequest):
    """
    Main machine learning prediction endpoint.
    Processes applicant data, scales features, and returns loan approval decision with analytics.
    """
    try:
        raw_dict = application.model_dump()
        result = predictor.predict(raw_dict)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the complete responsive CreditWise Web Application."""
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CreditWise AI | Smart Loan Approval & Risk Assessment</title>
  <!-- Tailwind CSS & Google Fonts -->
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/lucide@latest"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['"Plus Jakarta Sans"', 'sans-serif'],
          },
          colors: {
            brand: {
              50: '#eef2ff',
              100: '#e0e7ff',
              500: '#6366f1',
              600: '#4f46e5',
              700: '#4338ca',
              900: '#1e1b4b'
            }
          }
        }
      }
    }
  </script>
  <style>
    body {
      background: radial-gradient(circle at 50% 0%, #1e1e2f 0%, #0d0f17 100%);
      color: #e2e8f0;
      min-height: 100vh;
    }
    .glass-card {
      background: rgba(22, 27, 44, 0.7);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-input {
      background: rgba(15, 20, 35, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #f8fafc;
      transition: all 0.2s ease;
    }
    .glass-input:focus {
      border-color: #6366f1;
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
      outline: none;
    }
    .glow-approved {
      box-shadow: 0 0 35px -5px rgba(16, 185, 129, 0.35);
    }
    .glow-rejected {
      box-shadow: 0 0 35px -5px rgba(239, 68, 68, 0.35);
    }
  </style>
</head>
<body class="font-sans antialiased selection:bg-indigo-500 selection:text-white pb-16">

  <!-- Navigation Bar -->
  <nav class="border-b border-white/10 glass-card sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
          <i data-lucide="shield-check" class="w-6 h-6 text-white"></i>
        </div>
        <div>
          <span class="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-indigo-100 to-indigo-300 bg-clip-text text-transparent">CreditWise AI</span>
          <span class="hidden sm:inline-block text-xs px-2 py-0.5 ml-2 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium">ML v1.0</span>
        </div>
      </div>
      <div class="flex items-center space-x-4">
        <a href="/docs" target="_blank" class="text-xs font-semibold px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-all flex items-center space-x-1.5 border border-white/10">
          <i data-lucide="code" class="w-3.5 h-3.5"></i>
          <span>Swagger API Docs</span>
        </a>
        <div class="flex items-center space-x-2 text-xs font-medium text-emerald-400 bg-emerald-500/10 px-3 py-1.5 rounded-full border border-emerald-500/20">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>Engine Online</span>
        </div>
      </div>
    </div>
  </nav>

  <!-- Hero Header -->
  <header class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 pb-6 text-center sm:text-left">
    <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
      <div>
        <h1 class="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
          Intelligent Loan Eligibility Assessment
        </h1>
        <p class="text-slate-400 mt-2 text-sm sm:text-base max-w-2xl">
          Powered by a Gaussian Naive Bayes classifier & calibrated standard scaler. Evaluate applicant creditworthiness, risk probabilities, and automated decision rationales.
        </p>
      </div>
      <!-- Preset Buttons -->
      <div class="flex flex-wrap items-center gap-2 self-start sm:self-auto">
        <span class="text-xs text-slate-400 mr-1 font-medium">Quick Presets:</span>
        <button onclick="loadScenario('prime_approved')" class="text-xs font-medium px-3 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 transition flex items-center space-x-1">
          <i data-lucide="user-check" class="w-3.5 h-3.5"></i>
          <span>Prime Approved</span>
        </button>
        <button onclick="loadScenario('moderate_borderline')" class="text-xs font-medium px-3 py-1.5 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/30 transition flex items-center space-x-1">
          <i data-lucide="alert-circle" class="w-3.5 h-3.5"></i>
          <span>Moderate Risk</span>
        </button>
        <button onclick="loadScenario('high_risk_rejected')" class="text-xs font-medium px-3 py-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30 transition flex items-center space-x-1">
          <i data-lucide="user-x" class="w-3.5 h-3.5"></i>
          <span>High Risk (Reject)</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 lg:grid-cols-12 gap-8 mt-2">
    
    <!-- Application Input Form (7 Cols) -->
    <div class="lg:col-span-7 space-y-6">
      <form id="loanForm" onsubmit="handleFormSubmit(event)" class="space-y-6">
        
        <!-- Section 1: Demographics & Personal -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <div class="flex items-center space-x-2 text-indigo-400 mb-4 font-semibold text-sm">
            <i data-lucide="user" class="w-4 h-4"></i>
            <span>Applicant Demographics</span>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Age</label>
              <input type="number" id="Age" value="38" min="18" max="99" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Gender</label>
              <select id="Gender" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Male" selected>Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Marital Status</label>
              <select id="Marital_Status" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Married" selected>Married</option>
                <option value="Single">Single</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Dependents</label>
              <input type="number" id="Dependents" value="1" min="0" max="10" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Education Level</label>
              <select id="Education_Level" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Graduate" selected>Graduate</option>
                <option value="Not Graduate">Not Graduate</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Property Area</label>
              <select id="Property_Area" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Urban" selected>Urban</option>
                <option value="Semiurban">Semiurban</option>
                <option value="Rural">Rural</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Section 2: Employment & Income -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <div class="flex items-center space-x-2 text-indigo-400 mb-4 font-semibold text-sm">
            <i data-lucide="briefcase" class="w-4 h-4"></i>
            <span>Employment & Monthly Incomes</span>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Employment Status</label>
              <select id="Employment_Status" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Salaried" selected>Salaried</option>
                <option value="Self-employed">Self-employed</option>
                <option value="Contract">Contract</option>
                <option value="Unemployed">Unemployed</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Employer Category</label>
              <select id="Employer_Category" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Private" selected>Private</option>
                <option value="Government">Government</option>
                <option value="MNC">MNC</option>
                <option value="Business">Business</option>
                <option value="Unemployed">Unemployed</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Applicant Income ($/mo)</label>
              <input type="number" id="Applicant_Income" value="14500" min="0" step="100" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Coapplicant Income ($/mo)</label>
              <input type="number" id="Coapplicant_Income" value="4000" min="0" step="100" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
          </div>
        </div>

        <!-- Section 3: Financial Health & Loan Request -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <div class="flex items-center space-x-2 text-indigo-400 mb-4 font-semibold text-sm">
            <i data-lucide="credit-card" class="w-4 h-4"></i>
            <span>Credit History & Loan Details</span>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Credit Score (300-850)</label>
              <input type="number" id="Credit_Score" value="735" min="300" max="850" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">DTI Ratio (e.g. 0.25)</label>
              <input type="number" id="DTI_Ratio" value="0.25" min="0.0" max="1.0" step="0.01" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Existing Loans</label>
              <input type="number" id="Existing_Loans" value="1" min="0" max="10" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Liquid Savings ($)</label>
              <input type="number" id="Savings" value="16000" min="0" step="100" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Collateral Value ($)</label>
              <input type="number" id="Collateral_Value" value="38000" min="0" step="100" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Loan Purpose</label>
              <select id="Loan_Purpose" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="Home" selected>Home</option>
                <option value="Personal">Personal</option>
                <option value="Car">Car</option>
                <option value="Business">Business</option>
                <option value="Education">Education</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Requested Loan ($)</label>
              <input type="number" id="Loan_Amount" value="20000" min="500" step="500" required class="glass-input w-full px-3 py-2 rounded-xl text-sm">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Loan Term (Months)</label>
              <select id="Loan_Term" class="glass-input w-full px-3 py-2 rounded-xl text-sm">
                <option value="12">12 Months (1 yr)</option>
                <option value="24">24 Months (2 yrs)</option>
                <option value="36">36 Months (3 yrs)</option>
                <option value="48" selected>48 Months (4 yrs)</option>
                <option value="60">60 Months (5 yrs)</option>
                <option value="72">72 Months (6 yrs)</option>
                <option value="84">84 Months (7 yrs)</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Submit Button -->
        <button type="submit" id="submitBtn" class="w-full py-4 px-6 rounded-2xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-bold text-base shadow-lg shadow-indigo-600/30 transition-all flex items-center justify-center space-x-2">
          <i data-lucide="sparkles" class="w-5 h-5"></i>
          <span>Evaluate Loan Eligibility</span>
        </button>
      </form>
    </div>

    <!-- Prediction & Analysis Results Panel (5 Cols) -->
    <div class="lg:col-span-5 space-y-6">
      
      <!-- Primary Decision Card -->
      <div id="resultCard" class="glass-card rounded-2xl p-6 shadow-xl transition-all duration-300">
        <div class="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Automated Assessment</div>
        
        <!-- Decision Header -->
        <div class="flex items-center justify-between pb-6 border-b border-white/10">
          <div>
            <div id="decisionBadge" class="text-3xl font-extrabold tracking-tight text-white flex items-center space-x-2">
              <span>Ready for Review</span>
            </div>
            <div id="riskTierText" class="text-xs text-slate-400 mt-1 font-medium">Fill application form and evaluate</div>
          </div>
          <div id="decisionIcon" class="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center border border-white/10">
            <i data-lucide="calculator" class="w-7 h-7 text-indigo-400"></i>
          </div>
        </div>

        <!-- Probability Meter -->
        <div class="mt-6">
          <div class="flex justify-between items-center text-xs mb-1.5">
            <span class="text-slate-300 font-medium">Approval Probability</span>
            <span id="probPercentage" class="text-indigo-400 font-bold">--%</span>
          </div>
          <div class="w-full h-3 rounded-full bg-slate-800 overflow-hidden p-0.5 border border-white/5">
            <div id="probBar" class="h-full rounded-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-700" style="width: 0%"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>High Risk (0%)</span>
            <span>Threshold (50%)</span>
            <span>Prime (100%)</span>
          </div>
        </div>

        <!-- Key Financial Metrics Grid -->
        <div class="grid grid-cols-2 gap-3 mt-6">
          <div class="p-3 rounded-xl bg-white/5 border border-white/5">
            <div class="text-[11px] text-slate-400">Credit Score Tier</div>
            <div id="metricCreditTier" class="text-sm font-bold text-slate-200 mt-0.5">--</div>
          </div>
          <div class="p-3 rounded-xl bg-white/5 border border-white/5">
            <div class="text-[11px] text-slate-400">DTI Health</div>
            <div id="metricDti" class="text-sm font-bold text-slate-200 mt-0.5">--</div>
          </div>
          <div class="p-3 rounded-xl bg-white/5 border border-white/5">
            <div class="text-[11px] text-slate-400">Collateral Coverage</div>
            <div id="metricCoverage" class="text-sm font-bold text-slate-200 mt-0.5">--</div>
          </div>
          <div class="p-3 rounded-xl bg-white/5 border border-white/5">
            <div class="text-[11px] text-slate-400">Total Monthly Income</div>
            <div id="metricIncome" class="text-sm font-bold text-slate-200 mt-0.5">--</div>
          </div>
        </div>

        <!-- Insights Accordion / Lists -->
        <div class="mt-6 space-y-4">
          <!-- Strengths -->
          <div>
            <div class="text-xs font-semibold text-emerald-400 flex items-center space-x-1.5 mb-2">
              <i data-lucide="check-circle" class="w-3.5 h-3.5"></i>
              <span>Key Strengths</span>
            </div>
            <ul id="strengthsList" class="text-xs text-slate-300 space-y-1.5 list-disc pl-4">
              <li class="text-slate-500 italic list-none">Submit applicant profile to view strengths.</li>
            </ul>
          </div>

          <!-- Risk Factors -->
          <div>
            <div class="text-xs font-semibold text-rose-400 flex items-center space-x-1.5 mb-2">
              <i data-lucide="alert-triangle" class="w-3.5 h-3.5"></i>
              <span>Identified Risk Factors</span>
            </div>
            <ul id="risksList" class="text-xs text-slate-300 space-y-1.5 list-disc pl-4">
              <li class="text-slate-500 italic list-none">Submit applicant profile to view risk flags.</li>
            </ul>
          </div>

          <!-- Recommendations -->
          <div>
            <div class="text-xs font-semibold text-indigo-400 flex items-center space-x-1.5 mb-2">
              <i data-lucide="lightbulb" class="w-3.5 h-3.5"></i>
              <span>Advisor Recommendations</span>
            </div>
            <ul id="recsList" class="text-xs text-slate-300 space-y-1.5 list-disc pl-4">
              <li class="text-slate-500 italic list-none">Submit applicant profile to view recommendations.</li>
            </ul>
          </div>
        </div>

      </div>

      <!-- Architecture Card -->
      <div class="glass-card rounded-2xl p-5 border border-white/5 text-xs text-slate-400">
        <div class="font-semibold text-slate-200 flex items-center space-x-1.5 mb-2">
          <i data-lucide="cpu" class="w-4 h-4 text-indigo-400"></i>
          <span>Model Architecture Specs</span>
        </div>
        <p class="leading-relaxed">
          Gaussian Naive Bayes classifier trained on historical credit records with 27 standard-scaled features including non-linear interaction terms (<code class="text-indigo-300">Credit_Score²</code>, <code class="text-indigo-300">DTI_Ratio²</code>).
        </p>
      </div>

    </div>
  </main>

  <script>
    lucide.createIcons();

    // Sample Scenarios Cache
    let scenarios = null;

    async function fetchScenarios() {
      try {
        const res = await fetch('/api/scenarios');
        scenarios = await res.json();
      } catch (err) {
        console.error('Error loading scenarios:', err);
      }
    }
    fetchScenarios();

    function loadScenario(key) {
      if (!scenarios || !scenarios[key]) return;
      const data = scenarios[key].data;
      for (const [k, v] of Object.entries(data)) {
        const el = document.getElementById(k);
        if (el) {
          el.value = v;
        }
      }
      handleFormSubmit(new Event('submit'));
    }

    async function handleFormSubmit(e) {
      if (e) e.preventDefault();
      
      const btn = document.getElementById('submitBtn');
      const originalText = btn.innerHTML;
      btn.innerHTML = '<span class="animate-spin inline-block mr-2">&#8635;</span> Evaluating Model...';
      btn.disabled = true;

      const payload = {
        Applicant_Income: parseFloat(document.getElementById('Applicant_Income').value),
        Coapplicant_Income: parseFloat(document.getElementById('Coapplicant_Income').value),
        Age: parseFloat(document.getElementById('Age').value),
        Dependents: parseFloat(document.getElementById('Dependents').value),
        Existing_Loans: parseFloat(document.getElementById('Existing_Loans').value),
        Savings: parseFloat(document.getElementById('Savings').value),
        Collateral_Value: parseFloat(document.getElementById('Collateral_Value').value),
        Loan_Amount: parseFloat(document.getElementById('Loan_Amount').value),
        Loan_Term: parseFloat(document.getElementById('Loan_Term').value),
        Credit_Score: parseFloat(document.getElementById('Credit_Score').value),
        DTI_Ratio: parseFloat(document.getElementById('DTI_Ratio').value),
        Education_Level: document.getElementById('Education_Level').value,
        Employment_Status: document.getElementById('Employment_Status').value,
        Marital_Status: document.getElementById('Marital_Status').value,
        Loan_Purpose: document.getElementById('Loan_Purpose').value,
        Property_Area: document.getElementById('Property_Area').value,
        Gender: document.getElementById('Gender').value,
        Employer_Category: document.getElementById('Employer_Category').value
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const json = await res.json();
        if (json.success) {
          renderResult(json.data);
        } else {
          alert('Prediction failed: ' + (json.detail || 'Unknown error'));
        }
      } catch (err) {
        alert('Server connection error: ' + err.message);
      } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
        lucide.createIcons();
      }
    }

    function renderResult(data) {
      const card = document.getElementById('resultCard');
      const badge = document.getElementById('decisionBadge');
      const riskText = document.getElementById('riskTierText');
      const icon = document.getElementById('decisionIcon');
      const probText = document.getElementById('probPercentage');
      const probBar = document.getElementById('probBar');

      card.classList.remove('glow-approved', 'glow-rejected');

      if (data.loan_approved) {
        card.classList.add('glow-approved');
        badge.innerHTML = `<span class="text-emerald-400">APPROVED</span>`;
        icon.innerHTML = `<i data-lucide="check-check" class="w-8 h-8 text-emerald-400"></i>`;
        icon.className = 'w-14 h-14 rounded-2xl bg-emerald-500/20 flex items-center justify-center border border-emerald-500/40';
        probBar.className = 'h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-300 transition-all duration-700';
      } else {
        card.classList.add('glow-rejected');
        badge.innerHTML = `<span class="text-rose-400">REJECTED</span>`;
        icon.innerHTML = `<i data-lucide="x-circle" class="w-8 h-8 text-rose-400"></i>`;
        icon.className = 'w-14 h-14 rounded-2xl bg-rose-500/20 flex items-center justify-center border border-rose-500/40';
        probBar.className = 'h-full rounded-full bg-gradient-to-r from-rose-500 to-amber-500 transition-all duration-700';
      }

      riskText.textContent = `Confidence: ${data.approval_probability}% Approval • ${data.risk_tier}`;
      probText.textContent = `${data.approval_probability}%`;
      probBar.style.width = `${Math.min(Math.max(data.approval_probability, 3), 100)}%`;

      // Metrics
      document.getElementById('metricCreditTier').textContent = `${data.metrics.credit_score} (${data.metrics.credit_score_tier})`;
      document.getElementById('metricDti').textContent = `${data.metrics.dti_ratio} (${data.metrics.dti_status})`;
      document.getElementById('metricCoverage').textContent = `${data.metrics.collateral_coverage_ratio}x Loan Value`;
      document.getElementById('metricIncome').textContent = `$${data.metrics.total_monthly_income.toLocaleString()} / mo`;

      // Lists
      populateList('strengthsList', data.insights.strengths, 'No major strengths noted.');
      populateList('risksList', data.insights.risks, 'No high-risk flags identified.');
      populateList('recsList', data.insights.recommendations, 'Standard underwriting terms apply.');

      lucide.createIcons();
    }

    function populateList(elementId, items, emptyMessage) {
      const el = document.getElementById(elementId);
      el.innerHTML = '';
      if (!items || items.length === 0) {
        el.innerHTML = `<li class="text-slate-500 italic list-none">${emptyMessage}</li>`;
        return;
      }
      items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        el.appendChild(li);
      });
    }

    // Auto-run prime scenario on initial load
    window.addEventListener('DOMContentLoaded', () => {
      setTimeout(() => {
        loadScenario('prime_approved');
      }, 600);
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  🚀 CreditWise AI — Loan Approval Engine is LIVE!")
    print("  👉 Web Dashboard: http://127.0.0.1:8000")
    print("  👉 Swagger API:   http://127.0.0.1:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

lucide.createIcons();

const PRESETS = {
  prime: {
    Applicant_Income: 16000,
    Coapplicant_Income: 6000,
    Age: 42,
    Dependents: 1,
    Existing_Loans: 1,
    Savings: 18500,
    Collateral_Value: 45000,
    Loan_Amount: 15000,
    Loan_Term: 48,
    Credit_Score: 765,
    DTI_Ratio: 0.22,
    Education_Level: 'Graduate',
    Employment_Status: 'Salaried',
    Marital_Status: 'Married',
    Loan_Purpose: 'Home',
    Property_Area: 'Urban',
    Gender: 'Male',
    Employer_Category: 'MNC'
  },
  moderate: {
    Applicant_Income: 9500,
    Coapplicant_Income: 2500,
    Age: 33,
    Dependents: 2,
    Existing_Loans: 2,
    Savings: 8000,
    Collateral_Value: 18000,
    Loan_Amount: 20000,
    Loan_Term: 36,
    Credit_Score: 665,
    DTI_Ratio: 0.38,
    Education_Level: 'Graduate',
    Employment_Status: 'Self-employed',
    Marital_Status: 'Single',
    Loan_Purpose: 'Car',
    Property_Area: 'Semiurban',
    Gender: 'Female',
    Employer_Category: 'Private'
  },
  reject: {
    Applicant_Income: 4500,
    Coapplicant_Income: 0,
    Age: 26,
    Dependents: 3,
    Existing_Loans: 4,
    Savings: 1200,
    Collateral_Value: 5000,
    Loan_Amount: 32000,
    Loan_Term: 72,
    Credit_Score: 560,
    DTI_Ratio: 0.58,
    Education_Level: 'Not Graduate',
    Employment_Status: 'Unemployed',
    Marital_Status: 'Single',
    Loan_Purpose: 'Personal',
    Property_Area: 'Rural',
    Gender: 'Male',
    Employer_Category: 'Unemployed'
  }
};

function applyPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  for (const [k, v] of Object.entries(p)) {
    const el = document.getElementById(k);
    if (el) el.value = v;
  }
  document.getElementById('loanForm').dispatchEvent(new Event('submit'));
}

document.getElementById('loanForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('submitBtn');
  const origText = btn.innerHTML;
  btn.innerHTML = 'Evaluating Model...';
  btn.disabled = true;

  const baseUrl = document.getElementById('apiBaseUrl').value.replace(/\/+$/, '');

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
    const res = await fetch(`${baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await res.json();
    if (json.success) {
      renderResult(json.data);
    } else {
      alert('Prediction Error: ' + (json.detail || 'Unknown error'));
    }
  } catch (err) {
    alert('Failed to connect to backend at ' + baseUrl + '. Make sure the backend is running!\nError: ' + err.message);
  } finally {
    btn.innerHTML = origText;
    btn.disabled = false;
    lucide.createIcons();
  }
});

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
    probBar.className = 'h-full rounded-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-all duration-700';
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

# 🔍 Ariq Audit Review Insight Quality

<<<<<<< HEAD
AI Model Fairness & Bias Audit Tool — built for internal audit and model risk teams.

## Dashboard Screenshot
![Dashboard](reports/assets/Screenshot%202026-05-10%20170857.png)

## PDF Report Example
![Audit Report](reports/assets/audit_report.png)
=======
> A tool that audits machine learning models for bias, 
> fairness violations, and explainability gaps
> generating professional PDF audit reports with 
> risk-rated findings and remediation recommendations.

## Why this exists
As AI models are increasingly used in high-stakes 
decisions (credit, insurance, hiring), internal audit 
teams need tools to assess whether these models treat 
all demographic groups fairly. Ariq Audit Review Insight Quality 
simulates that audit process end-to-end.

## What it produces
- Demographic Parity Difference scores per sensitive attribute
- Equalized Odds analysis across race and gender groups
- SHAP-based explainability visualisations
- Risk-rated findings (Low / Medium / High / Critical)
- Downloadable PDF audit report with recommendations
- Interactive Streamlit dashboard for stakeholders
>>>>>>> 1b53c1a5563bb341fb5f6a6a940e280586558414

## Tech stack
Python · scikit-learn · fairlearn · SHAP · Streamlit · fpdf2 · pandas · matplotlib

## How to run
```bash
pip install -r requirements.txt
python src/train.py
streamlit run src/dashboard.py
<<<<<<< HEAD
=======

## Dashboard Screenshot
![Dashboard](reports/assets/Screenshot%202026-05-10%20170857.png)

## PDF Report Example
![Audit Report](reports/assets/audit_report.png)
>>>>>>> 1b53c1a5563bb341fb5f6a6a940e280586558414

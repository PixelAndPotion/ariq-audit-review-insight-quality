# 🔍 Ariq Audit Review Insight Quality

AI Model Fairness & Bias Audit Tool — built for internal audit and model risk teams.

## Dashboard Screenshot
![Dashboard](reports/assets/Screenshot%202026-05-10%20170857.png)

## PDF Report Example
![Audit Report](reports/assets/audit_report.png)

## Tech stack
Python · scikit-learn · fairlearn · SHAP · Streamlit · fpdf2 · pandas · matplotlib

## How to run
```bash
pip install -r requirements.txt
python src/train.py
streamlit run src/dashboard.py

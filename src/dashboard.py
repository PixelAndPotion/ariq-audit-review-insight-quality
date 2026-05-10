# src/dashboard.py
import streamlit as st
import pickle
import pandas as pd
from audit import run_bias_audit
from explainer import generate_shap_plot
from report import generate_report
import os

st.set_page_config(
    page_title="AriQ: Audit Review Insight Quality",
    page_icon="🔍",
    layout="wide"
)

# Header
st.markdown("""
    <h1 style='color:#C84B31'>🔍 AriQ: Audit Review Insight Quality</h1>
    <p style='color:#9A9188; font-size:1.1rem'>
    AI Model Fairness & Bias Audit Tool — 
    built for internal audit and model risk teams
    </p>
    <hr>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Model Type", "Random Forest")
col2.metric("Dataset", "Adult Income (UCI)")
col3.metric("Sensitive Attributes", "Sex, Race")

st.markdown("---")

if st.button("▶ Run Full Audit"):
    with st.spinner("Running bias audit..."):
        results = run_bias_audit()

    # Overall metrics
    st.subheader(" Model Performance")
    ov = results['overall']
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy",  f"{ov['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{ov['precision']*100:.1f}%")
    m3.metric("Recall",    f"{ov['recall']*100:.1f}%")

    # Findings
    st.subheader("⚠️ Bias Findings")
    risk_emoji = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢'}

    for f in results['findings']:
        risk = f['risk']
        with st.expander(
            f"{risk_emoji.get(risk,'⚪')} {f['attribute']} - Risk: {risk}"
        ):
            c1, c2 = st.columns(2)
            c1.metric("Demographic Parity Difference", f['dpd'])
            c2.metric("Equalized Odds Difference",     f['eod'])
            st.caption("**What this means:** A DPD above 0.1 indicates "
                      "the model selects outcomes at meaningfully different "
                      "rates across groups — a material fairness concern.")

    # SHAP
    st.subheader(" Model Explainability (SHAP)")
    generate_shap_plot()
    st.image('reports/assets/shap_summary.png',
             caption="Feature importance - what drives model decisions")

    # Download report
    st.subheader(" Audit Report")
    generate_report(results)
    with open('reports/audit_report.pdf', 'rb') as f:
        st.download_button(
            "⬇ Download Full PDF Audit Report",
            f, "Ariq_Audit_Review_Insight_Quality.pdf", "application/pdf"
        )
    st.success("Audit complete. Report ready for download.")

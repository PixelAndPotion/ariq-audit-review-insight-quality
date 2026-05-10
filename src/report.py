# src/report.py
from fpdf import FPDF
from datetime import datetime
import json

class AuditReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(200, 75, 49)
        self.cell(0, 8, 'FairSight AI Audit Report', align='R')
        self.ln(4)
        self.set_draw_color(200, 75, 49)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150)
        self.cell(0, 8,
            f'FairSight v1.0 - Confidential - Page {self.page_no()}',
            align='C')

def generate_report(results: dict, output_path='reports/audit_report.pdf'):
    pdf = AuditReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cover block
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(13, 13, 13)
    pdf.ln(8)
    pdf.cell(0, 10, 'AI MODEL BIAS & FAIRNESS AUDIT', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(100)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Model: Random Forest Classifier | Dataset: Adult Income (UCI)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Performance summary
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(13, 13, 13)
    pdf.cell(0, 8, '1. Model Performance Summary', new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(212, 206, 198)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    ov = results['overall']
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(74, 69, 64)
    for k, v in ov.items():
        pdf.cell(0, 7, f"  {k.capitalize()}: {v}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Findings
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(13, 13, 13)
    pdf.cell(0, 8, '2. Bias Audit Findings', new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    risk_colors = {
        'CRITICAL': (180, 0, 0),
        'HIGH':     (200, 75, 49),
        'MEDIUM':   (184, 149, 42),
        'LOW':      (45, 106, 79),
    }

    for i, f in enumerate(results['findings'], 1):
        risk = f['risk']
        r, g, b = risk_colors.get(risk, (0, 0, 0))

        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(13, 13, 13)
        pdf.cell(0, 8, f"  Finding {i}: Sensitive Attribute - {f['attribute']}", new_x="LMARGIN", new_y="NEXT")

        # Risk badge
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(28, 6, f"  {risk}", fill=True)
        pdf.ln(8)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(74, 69, 64)
        pdf.cell(0, 6, f"  Demographic Parity Difference (DPD): {f['dpd']}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"  Equalized Odds Difference (EOD): {f['eod']}", new_x="LMARGIN", new_y="NEXT")

        # Interpretation
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(100)
        interp = interpret_finding(f)
        pdf.multi_cell(0, 6, f"  Interpretation: {interp}")
        pdf.ln(4)

    # SHAP plot
    try:
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(13, 13, 13)
        pdf.cell(0, 8, '3. Model Explainability (SHAP)', new_x="LMARGIN", new_y="NEXT")
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.image('reports/assets/shap_summary.png', w=170)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(150)
        pdf.cell(0, 6,
            'Figure 1: SHAP feature importance - features pushing model decisions toward high/low income prediction.',
            new_x="LMARGIN", new_y="NEXT")
    except:
        pass

    # Recommendations
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(13, 13, 13)
    pdf.cell(0, 8, '4. Recommendations & Remediation', new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(74, 69, 64)
    recs = [
        "1. Apply fairness constraints during model retraining (e.g. ExponentiatedGradient).",
        "2. Remove or proxy-blind sensitive attributes from feature set where legally required.",
        "3. Implement ongoing fairness monitoring - re-audit model quarterly.",
        "4. Escalate CRITICAL/HIGH findings to model risk committee for sign-off.",
        "5. Document audit trail and findings in the model risk register.",
    ]
    for rec in recs:
        pdf.multi_cell(0, 6, f"  {rec}")
        pdf.ln(1)

    pdf.output(output_path)
    print(f"Report saved to {output_path}")

def interpret_finding(f):
    dpd = abs(f['dpd'])
    attr = f['attribute'].lower()
    if dpd > 0.2:
        return (f"Severe disparity detected across {attr} groups. "
                "Model outcomes differ significantly - this constitutes "
                "a material fairness risk requiring immediate remediation.")
    elif dpd > 0.1:
        return (f"Meaningful disparity across {attr}. "
                "Outcomes are not equitable - recommend retraining "
                "with fairness constraints applied.")
    elif dpd > 0.05:
        return (f"Moderate disparity across {attr}. "
                "Monitor closely and document as a known model limitation.")
    else:
        return (f"Acceptable parity across {attr} groups. "
                "Continue routine monitoring.")

if __name__ == "__main__":
    from audit import run_bias_audit
    results = run_bias_audit()
    generate_report(results)

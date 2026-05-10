# src/audit.py
import pickle
import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_model():
    return pickle.load(open('models/model.pkl', 'rb'))

def run_bias_audit():
    model, X_test, y_test, s_test = load_model()
    y_pred = model.predict(X_test)

    results = {}

    # ── Overall performance ──────────────────────────────────────
    results['overall'] = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred), 4),
    }

    # ── Fairness metrics per sensitive feature ───────────────────
    findings = []

    for col in ['sex', 'race']:
        sensitive_col = s_test[col]

        dpd = demographic_parity_difference(y_test, y_pred,
                                            sensitive_features=sensitive_col)
        eod = equalized_odds_difference(y_test, y_pred,
                                        sensitive_features=sensitive_col)

        mf = MetricFrame(
            metrics={'accuracy': accuracy_score,
                     'selection_rate': selection_rate},
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_col
        )

        # Risk rating logic
        dpd_abs = abs(dpd)
        if dpd_abs > 0.2:   risk = "CRITICAL"
        elif dpd_abs > 0.1: risk = "HIGH"
        elif dpd_abs > 0.05:risk = "MEDIUM"
        else:               risk = "LOW"

        findings.append({
            'attribute':   col.upper(),
            'dpd':         round(dpd, 4),   # demographic parity difference
            'eod':         round(eod, 4),   # equalized odds difference
            'risk':        risk,
            'group_stats': mf.by_group.to_dict()
        })

    results['findings'] = findings
    return results

if __name__ == "__main__":
    import json
    r = run_bias_audit()
    print(json.dumps(r, indent=2, default=str))
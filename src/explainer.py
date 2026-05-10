# src/explainer.py
import shap
import pickle
import matplotlib.pyplot as plt
import os

def generate_shap_plot():
    # Load the trained model and test data
    model, X_test, y_test, s_test = pickle.load(
        open('models/model.pkl', 'rb'))

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:200])  # sample for speed

    # Ensure reports/assets exists
    os.makedirs('reports/assets', exist_ok=True)

    # Summary plot
    plt.figure()

    # Handle binary classification (list of arrays) vs single array
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap.summary_plot(shap_values[1], X_test[:200],
                          show=False, max_display=10)
    else:
        shap.summary_plot(shap_values, X_test[:200],
                          show=False, max_display=10)

    plt.tight_layout()
    plt.savefig('reports/assets/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved.")

if __name__ == "__main__":
    generate_shap_plot()

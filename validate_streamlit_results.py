"""Compare reloaded model metrics against the saved Step 2 baseline; no fitting."""
from pathlib import Path
import joblib
import pandas as pd
from evaluate_models import evaluate_classifier

root = Path(__file__).resolve().parent
baseline = pd.read_csv(root / "results" / "model_comparison.csv").set_index("Model")
test = pd.read_csv(root / "test_data.csv")
metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
for name in baseline.index:
    model = joblib.load(root / "model" / f"{name}.joblib")
    X = test[list(model.feature_names_in_)]
    actual, _, _ = evaluate_classifier(model, X, test["Class"])
    differences = {m: abs(actual[m] - baseline.loc[name, m]) for m in metric_names}
    worst = max(differences.values())
    print(f"{name}: maximum metric difference = {worst:.10f}")
    if worst > 1e-6:
        raise AssertionError(f"Metric mismatch for {name}: {differences}")
print("PASS: all saved models reproduce Step 2 metrics without retraining.")

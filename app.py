"""BITS ML Assignment 2: educational Streamlit evaluation UI.

Loads previously fitted pipelines. Never fits/retrains a model.
Review and personalize the interface before using it in your own project.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from evaluate_models import evaluate_classifier

TARGET = "Class"
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Gaussian Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

st.set_page_config(page_title="Dry Bean Classification Lab", layout="wide")
st.title("Dry Bean Classification Lab")
st.caption("BITS ML Assignment 2 — evaluate saved models on labeled test data; no training occurs in this app")
st.info("Upload a CSV with all 16 feature columns and the Class target. Use only trusted model files.")

selected = st.selectbox("Choose a saved classifier", list(MODEL_NAMES))
upload = st.file_uploader("Upload labeled test CSV", type=["csv"])

if upload is None:
    st.write("Select a model and upload `test_data.csv` to display its metrics and confusion matrix.")
    st.stop()

try:
    data = pd.read_csv(upload)
except Exception as exc:
    st.error(f"Unable to read CSV: {exc}")
    st.stop()

if data.empty or TARGET not in data.columns:
    st.error("CSV must contain at least one row and a 'Class' target column.")
    st.stop()
if data.columns.duplicated().any():
    st.error("CSV contains duplicate column names.")
    st.stop()
if data.isna().any().any():
    st.error("CSV has missing values; check your data before evaluation.")
    st.stop()

model_path = MODEL_DIR / f"{MODEL_NAMES[selected]}.joblib"
if not model_path.is_file():
    st.error(f"Saved model not found: model/{model_path.name}. Copy your Step 2 model files here.")
    st.stop()

try:
    # joblib files are Python pickle-based: only load files you trained/trust.
    model = joblib.load(model_path)
    if not hasattr(model, "feature_names_in_"):
        st.error("Saved pipeline lacks feature_names_in_. Verify it was fitted on a pandas DataFrame.")
        st.stop()
    features = list(model.feature_names_in_)
    received = [c for c in data.columns if c != TARGET]
    missing = sorted(set(features) - set(received))
    unexpected = sorted(set(received) - set(features))
    if missing or unexpected:
        st.error(f"CSV schema mismatch. Missing: {missing}; unexpected: {unexpected}")
        st.stop()
    X = data[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="raise")
    y = data[TARGET]
    if y.isna().any() or not set(y.unique()).issubset(set(model.classes_)):
        st.error("Target contains missing or unknown classes.")
        st.stop()
    if y.nunique() < 2:
        st.error("AUC needs at least two represented classes. Use the full labeled test CSV.")
        st.stop()
    # The original full 7-class test file is recommended for comparable multiclass AUC.
    if set(y.unique()) != set(model.classes_):
        st.warning("Not all trained classes are represented. Multiclass AUC may be undefined for this subset.")
    metrics, report, matrix = evaluate_classifier(model, X, y)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.success(f"Evaluated {len(data):,} rows using the saved {selected} model (no retraining).")
cols = st.columns(3)
for i, (name, value) in enumerate(metrics.items()):
    cols[i % 3].metric(name, f"{value:.4f}")

st.subheader("Confusion matrix")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=matrix.to_numpy(), display_labels=list(model.classes_)).plot(
    ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
)
st.pyplot(fig)
plt.close(fig)

st.subheader("Classification report")
st.code(report, language="text")
with st.expander("Preview uploaded test data"):
    st.dataframe(data.head(10))

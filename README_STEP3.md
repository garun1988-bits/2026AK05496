# Step 3 — Streamlit learning-support starter

**Academic integrity:** This is a learning example, not a ready-to-submit assignment. Read the code, adapt the layout and explanations, test it on BITS Virtual Lab, and write your own README observations. Keep meaningful Git commits. Do not claim any unperformed work.

## What it does

- Loads one of five existing fitted `.joblib` pipelines (no `fit()` anywhere in the app).
- Accepts a labeled test CSV with the same 16 feature names and a `Class` target.
- Displays Accuracy, weighted one-vs-rest multiclass AUC, weighted Precision/Recall/F1, and MCC.
- Displays a confusion matrix and a classification report.
- Validates column names, missing values, numeric input, and labels.

## Required existing project files

Copy `app.py`, `evaluate_models.py`, and `validate_streamlit_results.py` into the **root** of your existing `ml_assignment_2_step1` project. You already have `evaluate_models.py` from Step 2; this is the same version. **Do not overwrite your trained models or data.**

Your existing root must contain:

```
app.py
train_models.py
validate_streamlit_results.py
evaluate_models.py
test_data.csv
model/logistic_regression.joblib
model/decision_tree.joblib
model/knn.joblib
model/naive_bayes.joblib
model/svm.joblib
results/model_comparison.csv
```

## BITS Virtual Lab commands

```bash
cd ~/Downloads/ml_assignment_2_step1
python3 -m pip install streamlit matplotlib
python3 validate_streamlit_results.py
python3 -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL Streamlit prints, typically `http://localhost:8501`, **inside the BITS Virtual Lab browser**. If the lab blocks local web ports, ask lab support for the supported preview or port-forwarding method. Do not expose the app publicly from the lab.

Upload your **existing** `test_data.csv`, select each of the five models, and compare the displayed metrics against `results/model_comparison.csv`. Take one genuine screenshot showing the BITS Lab environment and running app or terminal. Do not fabricate screenshots.

## Expected checks

- Five models load without `FileNotFoundError`.
- Uploaded test data has 2,709 rows and all seven classes, based on your observed Step 1 output.
- SVM accuracy should be near `0.921004`; Logistic Regression near `0.919528` (your actual Step 2 baseline).
- `validate_streamlit_results.py` should report PASS for all models.
- Invalid CSVs should display a useful error, not a crash.

## Why this is safe against leakage

Saved scikit-learn pipelines contain the already fitted scaling transformations. The app calls `predict` and `predict_proba` only, not `fit` or `fit_transform`. It uses `Class` only for evaluation, never as an input feature.

## Deployment preparation (after local verification)

1. Use your original `requirements.txt`, adding Streamlit, matplotlib, joblib, pandas, numpy, and scikit-learn if missing. **Pin the scikit-learn version to the version used when the `.joblib` files were saved.** Check it with `python3 -c "import sklearn; print(sklearn.__version__)"`. Verify Python compatibility on Streamlit Cloud.
2. Include all five `.joblib` files, `app.py`, `evaluate_models.py`, `test_data.csv`, your training scripts, and your own `README.md` in GitHub. Do not commit unrelated datasets or credentials.
3. Deploy from [Streamlit Community Cloud](https://streamlit.io/cloud) by selecting the GitHub repository, branch, and `app.py`.
4. Open the public app, upload the test CSV, check every model, and record the live URL.
5. Prepare the final **single PDF** in the required order: GitHub link, Streamlit link, genuine BITS Lab screenshot, and README content.

## Questions to answer in your own words

- Why is the multiclass AUC one-vs-rest and weighted?
- Why do LR/KNN/SVM use `StandardScaler`, but a decision tree does not need it?
- Why is the uploaded target excluded from model input?
- Why should the model and preprocessing be saved together?
- Why can joblib files be dangerous if downloaded from untrusted sources?
- What explains the accuracy/F1 trade-off between SVM and Logistic Regression in your actual results?

# Machine Learning Assignment 2 — Dry Bean Classification

**BITS ID:** 2026AK05496  
**GitHub repository:** https://github.com/garun1988-bits/2026AK05496  
**Live Streamlit application:** https://2026ak05496-mshxsbm9walbkck4o6hbwt.streamlit.app/

## Problem statement

Classify dry bean samples into seven bean varieties using sixteen numeric measurements derived from bean images. Compare five supervised classification algorithms on the same held-out test set using Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC). Deploy the saved classifiers in a Streamlit application that accepts a labelled CSV for evaluation without retraining.

## Dataset description

**Source:** UCI Machine Learning Repository — Dry Bean Dataset (dataset 602): https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The source dataset has **13,611 records**, **16 numerical input features**, and **7 target classes**. After removing **68 duplicate records**, **13,543** samples remained. A stratified 80/20 split yielded **10,834 training rows** and **2,709 test rows**. The target column is `Class`. The same test set was used for all five classifiers.

The 16 input features are Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, and ShapeFactor4.

## Project files

- `app.py` — Streamlit application for selecting a saved classifier and evaluating a labelled CSV.
- `requirements.txt` — Python dependencies.
- `test_data.csv` — held-out labelled test data.
- `data/dry_bean.csv` — source dataset; `data/train_data.csv` — training split.
- `model/` — five saved classifier artifacts (`.joblib`).
- `train_models.py`, `evaluate_models.py`, `prepare_data.py`, `inspect_dataset.py`, `download_dataset.py` — dataset preparation, training and evaluation scripts.
- `results/` — model comparison, confusion matrices and classification reports.
- `screenshots/` — genuine BITS Virtual Lab application screenshots.

## Running the application

From the project root, install dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose one of the five saved models, upload `test_data.csv` (including the `Class` target), and inspect the six metrics and confusion matrix. The application loads saved models; it does not retrain them on uploaded data. The public deployment is linked above.

## Model comparison — held-out test set (2,709 rows)

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9935 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| K-Nearest Neighbors | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Support Vector Machine | 0.9210 | 0.9933 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

The reported values are rounded to four decimal places. Accuracy and recall coincide here because recall is calculated using weighted averaging in this multiclass setting.

## Observations and interpretation

1. **Logistic Regression:** Accuracy 0.9195, F1 0.9197 and MCC 0.9028. Its AUC of 0.9935 is the highest of the five models, indicating strong class-discrimination performance under the reported multiclass AUC calculation.
2. **Decision Tree:** Accuracy 0.8955 and MCC 0.8737. It is interpretable but trails Logistic Regression, KNN and SVM on the held-out set; its AUC of 0.9357 is also lower.
3. **K-Nearest Neighbors:** Accuracy 0.9155 and F1 0.9157. Its results are close to Logistic Regression and SVM, with an AUC of 0.9811.
4. **Gaussian Naive Bayes:** Accuracy 0.7630, F1 0.7607 and MCC 0.7143. This is the weakest model on accuracy and F1, despite its AUC of 0.9644. The feature-independence assumption may limit classification performance; this is an interpretation, not a proven causal explanation.
5. **Support Vector Machine:** Accuracy 0.9210, F1 0.9211 and MCC 0.9045, the highest values among the evaluated models. Its AUC of 0.9933 is slightly below Logistic Regression.

**Overall conclusion:** Support Vector Machine is the best overall model on this test set based on Accuracy, F1 and MCC. Logistic Regression leads marginally on AUC. These conclusions apply to this dataset, split and reported evaluation settings; they do not establish superiority on every possible dataset.

## Evidence and reproducibility

Five screenshots taken inside the BITS Virtual Lab on 9 November 2026 show the local Streamlit application at `127.0.0.1:8501` evaluating the same 2,709-row labelled test CSV using each saved classifier, with six metrics and a confusion matrix. The GitHub repository contains the code, saved models, test data and evaluation results. The public Streamlit URL is provided above.

## Dataset acknowledgement and academic integrity

Dataset: *Dry Bean Dataset*, UCI Machine Learning Repository, dataset 602, https://doi.org/10.24432/C50S4B. Dataset authors: Murat Koklu and Ilker Ali Ozkan. Review and verify the scripts, evaluation choices, and explanations before submission; this README describes the reported project results and is not a substitute for understanding the implementation.

# ML Assignment 2 — Step 1 Starter (UCI Dry Bean)

This ZIP contains **Step 1 only**: dataset acquisition, inspection, and reproducible train/test preparation. It does **not** contain trained models, Streamlit app, live URLs, or a BITS Virtual Lab screenshot. Those must be completed in later steps by the student.

## Assignment requirements
- One UCI/Kaggle classification dataset, >=500 instances and >=12 input features.
- Train Logistic Regression, Decision Tree, KNN, Gaussian/Multinomial Naive Bayes, and SVM on the same dataset.
- Report Accuracy, AUC, Precision, Recall, F1 and MCC for each.
- Save models, create interactive Streamlit app for CSV test upload, model selection, metrics and confusion matrix/classification report. Evaluate uploaded test data without retraining.
- Submit one PDF in order: GitHub URL, live Streamlit URL, screenshot of execution on BITS Virtual Lab, README content.
- Follow academic integrity guidance; AI is for learning support, not direct copy/paste submission.

## Dataset
UCI Dry Bean (dataset ID 602): https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
13,611 instances, 16 numeric input features, 7 classes. Cite the dataset source and authors in the final README/report. Download requires internet access from BITS Virtual Lab; if restricted, download on an allowed machine and transfer the original CSV to `data/dry_bean.csv`.

## BITS Virtual Lab setup (Linux terminal)
```bash
cd ml_assignment_2_step1
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_dataset.py
python inspect_dataset.py
python prepare_data.py
```
For Windows, activate with `.venv\\Scripts\\activate` and use `python` as appropriate. If venv or network is blocked, use your lab's approved environment.

## Generated files
- `data/dry_bean.csv`: full downloaded data
- `data/train_data.csv`: deduplicated stratified 80% training partition
- `test_data.csv`: deduplicated stratified 20% test partition, including target `Class`

Exact counts after deduplication can differ from the original 13,611. The fixed `random_state=42` makes the split reproducible. Step 2 must train **only** on `data/train_data.csv`, then evaluate on `test_data.csv`. Preprocessing parameters (e.g. scaling) must be fit on training data only and saved with each classifier, ideally as a scikit-learn Pipeline.

## Lab screenshot
Take a genuine screenshot of successful execution inside BITS Virtual Lab. Place it in `screenshots/` for your final submission. No screenshot is included in this starter.

## Step 1 checklist
- [ ] Dataset downloaded and source cited
- [ ] Python environment verified on BITS Virtual Lab
- [ ] Missing values, duplicates and class balance inspected
- [ ] Stratified split generated
- [ ] Lab screenshot captured

## Academic integrity
Review, run, understand, adapt and document your own implementation. Do not submit this starter unmodified as your own work.

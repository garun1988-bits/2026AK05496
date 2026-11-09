# 2026AK05496 — BITS ML Assignment 2 recovery instructions

This project reconstructs the original learning-support source code from Steps 1, 2 and 3. Generated datasets, trained models, results, and your genuine Virtual Lab screenshots are **not** included. Do not present the old screenshots or results as newly executed work. Read, understand, test and personalize the code before submission.

## 1. Extract in BITS Virtual Lab

Download `2026AK05496_Complete_Recovery.zip` into `~/Downloads`. In terminal:

```bash
cd ~/Downloads
unzip 2026AK05496_Complete_Recovery.zip
cd 2026AK05496
pwd
```

If this folder already exists, back it up before unzipping.

## 2. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

If packages are already installed, this should be quick. Avoid changing system Python without permission.

## 3. Step 1 — Download and prepare dataset

```bash
python3 download_dataset.py
python3 inspect_dataset.py
python3 prepare_data.py
ls -lh data/ test_data.csv
```

The UCI Dry Bean dataset (ID 602) has 16 numerical input features and a `Class` target. The preparation script removes exact duplicates and uses a stratified split. Expected historical output: 13,611 original rows; 13,543 after removing 68 duplicates; 10,834 training and 2,709 test rows. These counts depend on the fetched dataset.

## 4. Step 2 — Train and evaluate five classifiers

```bash
python3 train_models.py
python3 check_saved_models.py
cat results/model_comparison.csv
```

This creates five joblib pipelines in `model/` and results in `results/`. StandardScaler preprocessing for applicable models is inside each saved pipeline. The metrics include accuracy, multiclass AUC, precision, recall, F1 and MCC. See `README_STEP2.md` for design and evaluation notes.

## 5. Step 3 — Validate and run Streamlit

```bash
python3 validate_streamlit_results.py
python3 -m streamlit run app.py --server.address 127.0.0.1
```

Open `http://localhost:8501` inside the BITS Virtual Lab browser. Upload the newly generated `test_data.csv`, choose each classifier, and verify the six metrics, confusion matrix and classification report. The app loads saved models; it does not retrain them. See `README_STEP3.md`.

## 6. Screenshot and repository

Capture at least one **new genuine screenshot** in the BITS Virtual Lab showing execution, and store it in `screenshots/`. A prior screenshot cannot be recovered from these ZIPs. Check the assignment's exact requirements and academic-integrity rules. If Git is not installed in the lab, use an approved installation route or GitHub's web upload UI. Create repository `2026AK05496` only if that naming matches your assignment instructions; don't invent historical commits.

## 7. Deploy

Push `app.py`, `evaluate_models.py`, `requirements.txt`, `test_data.csv`, all five `model/*.joblib` files and other required project files to GitHub. In Streamlit Community Cloud select repository `2026AK05496`, branch and `app.py` as entry point. Test the deployed URL with the CSV. Review dependencies for compatibility before deployment.

## 8. Submission PDF

Use the assignment's required order: GitHub URL, deployed Streamlit URL, genuine BITS Lab screenshot, then complete README content. Create the PDF only after both links work. Do not submit unverified placeholder links.

## Prior run reference (not a substitute for new output)

Previous run reported SVM accuracy 0.921004, Logistic Regression 0.919528, KNN 0.915467, Decision Tree 0.895533, and Gaussian Naive Bayes 0.763012. Your new run should be verified independently.

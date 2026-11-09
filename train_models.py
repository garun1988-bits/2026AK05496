"""Learning example: fit five classifiers using the existing Step 1 training CSV.

Read README_STEP2.md and customize choices before using in your own project.
Run from the project root: python train_models.py
"""
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from evaluate_models import evaluate_classifier

TARGET = 'Class'
TRAIN_PATH = Path('data/train_data.csv')
TEST_PATH = Path('test_data.csv')
MODEL_DIR = Path('model')
RESULTS_DIR = Path('results')


def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError('Run Step 1 first. Expected data/train_data.csv and test_data.csv')
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if TARGET not in train or TARGET not in test:
        raise ValueError(f'Missing target column: {TARGET}')
    features = [c for c in train.columns if c != TARGET]
    if len(features) < 12 or len(train) < 500:
        raise ValueError('Training data does not meet assignment minimum size')
    if set(test.columns) != set(train.columns):
        raise ValueError('Test and training columns do not match')
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError('Missing feature values; decide how to handle them using training-only fitting')
    if set(y_test) - set(y_train):
        raise ValueError('Test contains a class absent from training')
    overlap = pd.merge(train.drop_duplicates(), test.drop_duplicates(), how='inner', on=list(train.columns))
    if not overlap.empty:
        raise ValueError(f'{len(overlap)} exact records appear in both train and test; inspect leakage')

    models = {
        'logistic_regression': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=2000, random_state=42))]),
        'decision_tree': Pipeline([('classifier', DecisionTreeClassifier(random_state=42))]),
        'knn': Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier(n_neighbors=5))]),
        'naive_bayes': Pipeline([('classifier', GaussianNB())]),
        'svm': Pipeline([('scaler', StandardScaler()), ('classifier', SVC(kernel='rbf', probability=True, random_state=42))]),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    records = []
    for name, model in models.items():
        print(f'\nTraining {name} ...', flush=True)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        metrics, report, matrix = evaluate_classifier(model, X_test, y_test)
        elapsed = time.perf_counter() - start
        joblib.dump(model, MODEL_DIR / f'{name}.joblib')
        (RESULTS_DIR / f'{name}_classification_report.txt').write_text(report, encoding='utf-8')
        matrix.to_csv(RESULTS_DIR / f'{name}_confusion_matrix.csv')
        records.append({'Model': name, **metrics, 'Training_and_evaluation_seconds': round(elapsed, 2)})
        print({k: round(v, 4) for k, v in metrics.items()})
        print(f'Saved model/{name}.joblib')
    comparison = pd.DataFrame(records)
    comparison.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    print('\nCOMPARISON (results/model_comparison.csv):')
    print(comparison.to_string(index=False))
    print('\nReview README_STEP2.md for interpretation and next steps.')


if __name__ == '__main__':
    main()

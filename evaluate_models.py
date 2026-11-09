"""Six mandatory metrics; weighted multiclass averaging and one-vs-rest AUC.

AUC uses predict_proba, not hard class predictions. This file is also reusable
in the later Streamlit application (saved models are loaded, never refit).
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)


def evaluate_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_
    if len(classes) > 2:
        auc = roc_auc_score(y_test, probabilities, labels=classes,
                            multi_class='ovr', average='weighted')
        average = 'weighted'
    else:
        auc = roc_auc_score(y_test, probabilities[:, 1], labels=classes)
        average = 'binary'
    metrics = {
        'Accuracy': accuracy_score(y_test, predictions),
        'AUC': auc,
        'Precision': precision_score(y_test, predictions, average=average, zero_division=0),
        'Recall': recall_score(y_test, predictions, average=average, zero_division=0),
        'F1': f1_score(y_test, predictions, average=average, zero_division=0),
        'MCC': matthews_corrcoef(y_test, predictions),
    }
    report = classification_report(y_test, predictions, labels=classes, zero_division=0)
    matrix = pd.DataFrame(confusion_matrix(y_test, predictions, labels=classes),
                          index=[f'True_{c}' for c in classes],
                          columns=[f'Pred_{c}' for c in classes])
    return metrics, report, matrix

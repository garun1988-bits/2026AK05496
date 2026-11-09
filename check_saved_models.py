"""Sanity check that persisted pipelines predict without fitting again."""
from pathlib import Path
import joblib
import pandas as pd

frame = pd.read_csv('test_data.csv')
X = frame.drop(columns=['Class'])
for path in sorted(Path('model').glob('*.joblib')):
    model = joblib.load(path)  # Load only files you created/trust.
    pred = model.predict(X.iloc[:5])
    print(f'{path.name}: first five predictions = {pred.tolist()}')

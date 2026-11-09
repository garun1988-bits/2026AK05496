from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parent / 'data' / 'dry_bean.csv'
if not path.exists():
    raise FileNotFoundError('Run download_dataset.py first')
df = pd.read_csv(path)
print('Rows and columns:', df.shape)
print('\nColumns:', df.columns.tolist())
print('\nFirst five rows:\n', df.head())
print('\nMissing values:\n', df.isnull().sum())
print('\nDuplicate rows:', df.duplicated().sum())
print('\nTarget distribution:\n', df['Class'].value_counts())
print('\nNumeric summary:\n', df.describe())
assert len(df) >= 500, 'At least 500 instances required'
assert 'Class' in df.columns, 'Class target missing'
assert df.drop(columns=['Class']).shape[1] >= 12, 'At least 12 input features required'
assert df['Class'].nunique() >= 2, 'Classification target needs at least two classes'
print('\nPASS: Dataset meets minimum size and classification requirements.')

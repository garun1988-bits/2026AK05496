from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parent
df = pd.read_csv(root / 'data' / 'dry_bean.csv')
print('Original rows:', len(df))
print('Duplicate rows:', df.duplicated().sum())
# Document the policy: remove exact duplicate rows before splitting to avoid leakage.
df = df.drop_duplicates().reset_index(drop=True)
if df.isnull().any().any():
    raise ValueError('Missing values found; decide on imputation during model pipeline design')
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# Save both partitions so Step 2 uses exactly the same split and avoids leakage.
train = X_train.copy()
train['Class'] = y_train
train.to_csv(root / 'data' / 'train_data.csv', index=False)
test = X_test.copy()
test['Class'] = y_test
test.to_csv(root / 'test_data.csv', index=False)
print('Rows after deduplication:', len(df))
print('Training rows:', len(train))
print('Testing rows:', len(test))
print('Features:', X.shape[1])
print('Saved data/train_data.csv and test_data.csv')

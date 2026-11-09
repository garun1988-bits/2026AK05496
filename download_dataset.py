from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
dataset = fetch_ucirepo(id=602)
X = dataset.data.features
y = dataset.data.targets
if X is None or y is None:
    raise RuntimeError('UCI repository did not return expected features and target')
df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
if 'Class' not in df.columns:
    raise ValueError(f'Expected Class target; got {list(df.columns)}')
output = DATA / 'dry_bean.csv'
df.to_csv(output, index=False)
print(f'Saved {output}')
print('Dataset shape:', df.shape)
print('Feature count:', X.shape[1])
print('Target classes:', df['Class'].nunique())

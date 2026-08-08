import os, sys
os.environ.setdefault("PYTHONPYCACHEPREFIX","/tmp/pyc")
from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
OUT=str(BASE/"data"/"ai4i2020.csv")
if os.path.exists(OUT):
    df=pd.read_csv(OUT); print("cached", df.shape); sys.exit(0)
from sklearn.datasets import fetch_openml
d=fetch_openml(data_id=42890, as_frame=True, parser="auto")
df=d.frame
print(df.shape); print(list(df.columns))
df.to_csv(OUT, index=False)
print("saved")


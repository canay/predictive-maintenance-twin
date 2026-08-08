import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

T0 = time.time(); MAX_SEC = 28
OUT = BASE + "/results/shap"
df = load_ai4i(); X = df[FEATS].values
for t in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
    for seed in range(5):
        fp = f"{OUT}/p_{t}_{seed}.json"
        if os.path.exists(fp):
            continue
        if time.time() - T0 > MAX_SEC:
            print("BUDGET"); sys.exit(2)
        y = df[t].values.astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                              random_state=seed)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    class_weight="balanced", n_jobs=2,
                                    random_state=seed).fit(Xtr, ytr)
        v = shap.TreeExplainer(rf).shap_values(X[y == 1])
        if isinstance(v, list):
            v = v[1]
        elif v.ndim == 3:
            v = v[..., 1]
        imp = np.abs(v).mean(0)
        jdump(dict(target=t, seed=seed, imp_posall=imp.tolist(), n=int(y.sum())), fp)
        print(t, seed)
print("ALL DONE")


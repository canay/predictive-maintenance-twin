import os, sys, json, glob, platform
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

S = {}
# risk layer
risk = {}
for m in ["LR", "RF", "HGB"]:
    risk[m] = {}
    for t in ["Machine failure"] + ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        fs = glob.glob(BASE + f"/results/risk/u_{m}_{t.replace(' ', '_')}_*.json")
        rs = [json.load(open(f)) for f in fs]
        risk[m][t] = {k: dict(mean=float(np.mean([r[k] for r in rs])),
                              sd=float(np.std([r[k] for r in rs])))
                      for k in ["auroc", "ap", "f1", "ece"]}
        risk[m][t]["n_pos_test"] = rs[0]["n_pos_test"]
S["risk"] = risk
# calibration
S["calibration"] = json.load(open(BASE + "/results/calibration.json"))
S["sim_params_keys"] = {k: v for k, v in json.load(open(BASE + "/results/sim_params.json")).items()
                        if k not in ("inc_vals", "inc_probs")}
sp = json.load(open(BASE + "/results/sim_params.json"))
S["mean_wear_increment"] = float(np.dot(sp["inc_vals"], sp["inc_probs"]))
# rul
rul = {}
for m in ["HGBR", "MLP"]:
    rs = [json.load(open(f)) for f in sorted(glob.glob(BASE + f"/results/rul/u2_{m}_*.json"))]
    rul[m] = dict(
        rmse=dict(mean=float(np.mean([r["rmse"] for r in rs])), sd=float(np.std([r["rmse"] for r in rs]))),
        mae=dict(mean=float(np.mean([r["mae"] for r in rs])), sd=float(np.std([r["mae"] for r in rs]))),
        prog_horizon_rel=dict(mean=float(np.mean([r["prog_horizon_frac"] for r in rs])),
                              sd=float(np.std([r["prog_horizon_frac"] for r in rs]))),
        ph02=dict(mean=float(np.mean([r["ph02_frac"] for r in rs])),
                  sd=float(np.std([r["ph02_frac"] for r in rs]))),
        alpha_lambda={k: dict(mean=float(np.mean([r["alpha_lambda"][k] for r in rs])),
                              sd=float(np.std([r["alpha_lambda"][k] for r in rs])))
                      for k in rs[0]["alpha_lambda"]})
S["rul"] = rul
# policy
S["policy"] = json.load(open(BASE + "/results/policy_eval.json"))
# shap
S["shap"] = json.load(open(BASE + "/results/shap_summary.json"))
jdump(S, BASE + "/results/results_summary.json")
# versions
import sklearn, pandas, scipy, matplotlib, shap as shp, torch, joblib
with open(BASE + "/results/versions.txt", "w") as f:
    f.write("python %s\n" % platform.python_version())
    for mod in [np, pandas, sklearn, scipy, matplotlib, shp, torch, joblib]:
        f.write("%s %s\n" % (mod.__name__, mod.__version__))
    f.write("platform %s\n" % platform.platform())
    f.write("dataset OpenML data_id=42890 ai4i2020 (10000 x 14)\n")
print("summary written")


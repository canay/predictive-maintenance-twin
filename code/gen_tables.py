import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
S=json.load(open(BASE+"/results/results_summary.json"))
def ms(d,f=3): return "%.*f $\\pm$ %.*f"%(f,d["mean"],f,d["sd"])
print("==== T1 risk ====")
for t in ["Machine failure","TWF","HDF","PWF","OSF","RNF"]:
    for m in ["LR","RF","HGB"]:
        d=S["risk"][m][t]
        print("%s & %s & %s & %s & %s & %s \\\\"%(t if m=="LR" else "",m,ms(d["auroc"]),ms(d["ap"]),ms(d["f1"]),ms(d["ece"])))
print("==== T2 calibration ====")
C=S["calibration"]
nm={"air":"Air temperature [K]","proc":"Process temperature [K]","rpm":"Rotational speed [rpm]","tq":"Torque [Nm]","wear":"Tool wear [min]"}
for k in ["air","proc","rpm","tq","wear"]:
    c=C[k]
    print("%s & %.2f / %.2f & %.2f / %.2f & %.3f & %.3f \\\\"%(nm[k],c["real_mu"],c["real_sd"],c["sim_mu"],c["sim_sd"],c["ks"],c["wass"]))
print("modes real/sim per1000:",{m:(C["failure_rates_real_per_1000_rows"][m],round(C["failure_rates_per_1000_cycles"][m],2)) for m in ["TWF","HDF","PWF","OSF","RNF"]})
print("any fail real %.1f sim %.1f"%(C["any_failure_rate_real"],C["any_failure_rate_sim"]))
print("==== T3 rul ====")
for m in ["HGBR","MLP"]:
    d=S["rul"][m]
    print("%s & %s & %s & %s & %s & %s & %s \\\\"%(m,ms(d["rmse"],2),ms(d["mae"],2),
      ms(d["alpha_lambda"]["0.3"]),ms(d["alpha_lambda"]["0.5"]),ms(d["alpha_lambda"]["0.9"]),ms(d["ph02"])))
# all-modes RUL for remark
fs=glob.glob(BASE+"/results/rul_allmodes/u_HGBR_*.json")
rs=[json.load(open(f)) for f in fs]
print("allmodes HGBR rmse %.2f+-%.2f mae %.2f"%(np.mean([r["rmse"] for r in rs]),np.std([r["rmse"] for r in rs]),np.mean([r["mae"] for r in rs])))
# degradation-only episode length
Ls=[]
for s in range(5):
    z=np.load(BASE+f"/data/pool2_dte_{s}.npz",allow_pickle=True)
    eid=z["eid"]; Ls+= list(np.bincount(eid.astype(int)))
print("deg-only mean episode len %.1f sd %.1f"%(np.mean(Ls),np.std(Ls)))
print("mean wear inc %.3f"%S["mean_wear_increment"])
print("==== T4 policy ====")
P=S["policy"]
EVAL_SEEDS=P.get("eval_seeds", list(range(5)))
names={"reactive":"Reactive","periodic":"Periodic","risk":"Risk-only","rul":"RUL-aware","risk_gated":"Risk-only + gate","rul_gated":"RUL-aware + gate"}
for r in ["2","5","10","20"]:
    for p in ["reactive","periodic","risk","rul","risk_gated","rul_gated"]:
        d=P["per_ratio"][r][p]; m=d["mean"]; sd=d["sd"]
        par={"reactive":"--","periodic":"$K{=}%d$"%d["param"] if p=="periodic" else "","risk":"$\\tau{=}%.2f$"%d["param"] if p=="risk" else ""}
        if p=="reactive": pa="--"
        elif p=="periodic": pa="$K{=}%d$"%d["param"]
        elif p.startswith("risk"): pa="$\\tau{=}%.2f$"%d["param"]
        else: pa="$h{=}%d$"%d["param"]
        print("%s & %s & %s & %.4f $\\pm$ %.4f & %.4f $\\pm$ %.4f & %.2f & %.1f & %.3f \\\\"%(
          r if p=="reactive" else "",names[p],pa,m["cost_rate"],sd["cost_rate"],m["avail"],sd["avail"],m["fails_per_1000"],m["n_prev"],m["false30_rate"]))
print("==== tests ====")
print(json.dumps(P["tests"],indent=1))
print("selection",P["selection"])
print("==== T6 attribution ====")
A=S["shap"]
for t in ["TWF","HDF","PWF","OSF","RNF"]:
    d=A[t]
    hm="%.2f $\\pm$ %.2f"%(d["hit_mean"],np.std(d["hit_rates"])) if d["hit_mean"] is not None else "--"
    print("%s & %s & %s & %s & %.2f $\\pm$ %.2f & %.3f $\\pm$ %.3f \\\\"%(t,", ".join(d["gt"]) if d["gt"] else "none (random)",d["n_pos"],hm,d["jaccard_top3"],d["jaccard_sd"],d["spearman"],d["spearman_sd"]))
print("gate pass rates:")
for s in EVAL_SEEDS:
    z=np.load(BASE+f"/results/gate_pe_{s}.npz"); print(" seed",s,round(float(z["gate"].mean()),3),len(z["cand"]))


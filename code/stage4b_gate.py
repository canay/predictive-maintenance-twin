import os, sys, time, numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
T0=time.time(); MAX_SEC=900
DD=BASE+"/data"; OUT=BASE+"/results"; os.makedirs(OUT,exist_ok=True)
EVAL_SEEDS=list(range(15))
MODEL_SEEDS=list(range(5))
import shap
WEAR=5
def sv_pos(ex,X):
    v=ex.shap_values(X)
    if isinstance(v,list): v=v[1]
    elif v.ndim==3: v=v[...,1]
    return v
def do_pool(tag,seed,model_seed):
    fp=f"{OUT}/gate_{tag}_{seed}.npz"
    if os.path.exists(fp): return True
    if time.time()-T0>MAX_SEC: return False
    z=np.load(f"{DD}/pool2_{tag}_{seed}.npz",allow_pickle=True)
    X=z["X"]
    rul=joblib.load(f"{DD}/rulmodel2_HGBR_{model_seed}.joblib").predict(X)
    risk=joblib.load(f"{DD}/riskrf_{model_seed}.joblib").predict_proba(X)[:,1]
    cand=np.where((rul<=16)|(risk>=0.5))[0]
    tops=[]
    for g in range(3):
        ex=shap.TreeExplainer(joblib.load(f"{DD}/gaterf_{g}.joblib"))
        v=np.abs(sv_pos(ex,X[cand]))
        top2=np.argsort(-v,axis=1)[:,:2]
        tops.append(np.any(top2==WEAR,axis=1))
    tops=np.vstack(tops)
    gate=tops.sum(0)>=2
    np.savez_compressed(fp,cand=cand,gate=gate,rul=rul,risk=risk,votes=tops)
    print("gate",tag,seed,"cand",len(cand),"passrate",float(gate.mean()))
    return True
jobs=[("pv",0,0)]+[("pe",s,s % len(MODEL_SEEDS)) for s in EVAL_SEEDS]
for tag,seed,ms in jobs:
    if not do_pool(tag,seed,ms): print("BUDGET"); sys.exit(2)
print("ALL DONE")


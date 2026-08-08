import os, sys, time, numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from simulator import fit_params, fit_pools, gen_pool
from sklearn.ensemble import RandomForestClassifier
T0=time.time(); MAX_SEC=120
DD=BASE+"/data"
EVAL_SEEDS=list(range(15))
df=load_ai4i(); P=fit_params(df); POOLS=fit_pools(df)
X=df[FEATS].values; y=df["Machine failure"].values.astype(int)
def step(name,fn):
    if os.path.exists(name): return
    if time.time()-T0>MAX_SEC: print("BUDGET"); sys.exit(2)
    fn(); print("done",os.path.basename(name))
for s in range(5):
    fp=f"{DD}/riskrf_{s}.joblib"
    step(fp,lambda fp=fp,s=s: joblib.dump(RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",n_jobs=2,random_state=s).fit(X,y),fp))
for g in range(3):
    fp=f"{DD}/gaterf_{g}.joblib"
    step(fp,lambda fp=fp,g=g: joblib.dump(RandomForestClassifier(n_estimators=100,min_samples_leaf=2,class_weight="balanced",n_jobs=2,random_state=100+g).fit(X,y),fp))
def gen_eval(tag,seed,n):
    fp=f"{DD}/pool2_{tag}_{seed}.npz"
    def fn():
        eps=gen_pool({"pe":7000,"pv":7900}[tag]+seed,n,P,POOLS,shocks=True)
        Xs=[e["X"] for e in eps]
        np.savez_compressed(fp,X=np.vstack(Xs),rul=np.concatenate([e["rul"] for e in eps]),
            eid=np.concatenate([np.full(e["T"],i) for i,e in enumerate(eps)]),
            modes=np.array([e["fail_mode"] for e in eps]),
            wtwf=np.array([e["w_twf"] for e in eps]),
            Ts=np.array([e["T"] for e in eps]))
    step(fp,fn)
for s in EVAL_SEEDS: gen_eval("pe",s,400)
gen_eval("pv",0,300)
print("ALL DONE")


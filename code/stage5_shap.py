import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
T0=time.time(); MAX_SEC=28
OUT=BASE+"/results/shap"; os.makedirs(OUT,exist_ok=True)
df=load_ai4i(); X=df[FEATS].values
GT={"TWF":[5],"HDF":[1,2,3],"PWF":[3,4],"OSF":[5,4,0],"RNF":[]}
def sv_pos(ex,Xs):
    v=ex.shap_values(Xs)
    if isinstance(v,list): v=v[1]
    elif v.ndim==3: v=v[...,1]
    return v
for target in ["TWF","HDF","PWF","OSF","RNF","Machine failure"]:
    for seed in range(5):
        fp=f"{OUT}/u_{target.replace(' ','_')}_{seed}.json"
        if os.path.exists(fp): continue
        if time.time()-T0>MAX_SEC: print("BUDGET"); sys.exit(2)
        y=df[target].values.astype(int)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=seed)
        rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",n_jobs=2,random_state=seed).fit(Xtr,ytr)
        rng=np.random.default_rng(seed)
        pos=np.where(yte==1)[0]; neg=rng.choice(np.where(yte==0)[0],size=min(800,(yte==0).sum()),replace=False)
        idx=np.concatenate([pos,neg])
        v=np.abs(sv_pos(shap.TreeExplainer(rf),Xte[idx]))
        imp=v.mean(0); imp_pos=v[:len(pos)].mean(0) if len(pos)>0 else imp
        gt=GT.get(target,[])
        k=len(gt)
        topk=list(np.argsort(-imp_pos)[:k]) if k>0 else []
        hit=len(set(topk)&set(gt))/k if k>0 else None
        jdump(dict(target=target,seed=seed,imp=imp.tolist(),imp_pos=imp_pos.tolist(),
            topk=[int(t) for t in topk],gt=gt,hit_rate=hit,
            top3=[int(t) for t in np.argsort(-imp_pos)[:3]],n_pos=int(len(pos))),fp)
        print(target,seed,"hit",hit)
print("ALL DONE")


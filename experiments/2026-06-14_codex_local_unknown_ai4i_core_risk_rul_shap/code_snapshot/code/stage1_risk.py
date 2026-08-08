import os, sys, time, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
T0=time.time(); MAX_SEC=32
OUT=BASE+"/results/risk"; os.makedirs(OUT,exist_ok=True)
df=load_ai4i(); X=df[FEATS].values
def make(model,seed):
    if model=="LR": return LogisticRegression(max_iter=2000,class_weight="balanced",C=1.0,random_state=seed),True
    if model=="RF": return RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",n_jobs=2,random_state=seed),False
    return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.1,early_stopping=True,random_state=seed),False
units=list(itertools.product(["LR","RF","HGB"],TARGETS,range(5)))
done=0
for model,target,seed in units:
    fp=f"{OUT}/u_{model}_{target.replace(' ','_')}_{seed}.json"
    if os.path.exists(fp): done+=1; continue
    if time.time()-T0>MAX_SEC: print(f"BUDGET done={done}/{len(units)}"); sys.exit(2)
    y=df[target].values.astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=seed)
    clf,scale=make(model,seed)
    if scale:
        sc=StandardScaler().fit(Xtr); Xtr2,Xte2=sc.transform(Xtr),sc.transform(Xte)
    else: Xtr2,Xte2=Xtr,Xte
    if model=="HGB":
        w=np.where(ytr==1,(ytr==0).sum()/max((ytr==1).sum(),1),1.0)
        clf.fit(Xtr2,ytr,sample_weight=w)
    else:
        clf.fit(Xtr2,ytr)
    p=clf.predict_proba(Xte2)[:,1]
    res=dict(model=model,target=target,seed=seed,
        auroc=float(roc_auc_score(yte,p)) if yte.sum()>0 else None,
        ap=float(average_precision_score(yte,p)),
        f1=float(f1_score(yte,(p>=0.5).astype(int),zero_division=0)),
        ece=ece(yte,p), n_pos_test=int(yte.sum()))
    jdump(res,fp); done+=1
print(f"ALL DONE {done}/{len(units)}")


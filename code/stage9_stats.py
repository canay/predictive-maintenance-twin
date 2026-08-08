import os, sys, time, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
T0=time.time(); MAX_SEC=34
OUT=BASE+"/results"
# --- 1. recompute all-modes RUL (legacy files corrupted), checkpointed per unit
def valid(fp):
    try: json.load(open(fp)); return True
    except Exception: return False
for model in ["HGBR","MLP"]:
    for s in range(5):
        fp=f"{OUT}/rul_allmodes/u_{model}_{s}.json"
        if valid(fp): continue
        if time.time()-T0>MAX_SEC: print("BUDGET"); sys.exit(2)
        ztr=np.load(BASE+f"/data/pool_tr_{s}.npz",allow_pickle=True)
        zte=np.load(BASE+f"/data/pool_te_{s}.npz",allow_pickle=True)
        Xtr,rtr=ztr["X"],ztr["rul"]; Xte,rte=zte["X"],zte["rul"]
        if model=="HGBR":
            from sklearn.ensemble import HistGradientBoostingRegressor
            m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.08,random_state=s)
            m.fit(Xtr,rtr); pred=m.predict(Xte)
        else:
            import torch, torch.nn as nn
            torch.manual_seed(s); torch.set_num_threads(2)
            mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9; rm,rs=rtr.mean(),rtr.std()
            Xt=torch.tensor((Xtr-mu)/sd,dtype=torch.float32); yt=torch.tensor((rtr-rm)/rs,dtype=torch.float32)[:,None]
            net=nn.Sequential(nn.Linear(Xtr.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
            opt=torch.optim.Adam(net.parameters(),lr=1e-3); lo=nn.MSELoss()
            n=len(Xt); idx=np.arange(n)
            for ep in range(60):
                np.random.default_rng(s*100+ep).shuffle(idx)
                for i in range(0,n,256):
                    b=idx[i:i+256]; opt.zero_grad(); l=lo(net(Xt[b]),yt[b]); l.backward(); opt.step()
            with torch.no_grad():
                pred=(net(torch.tensor((Xte-mu)/sd,dtype=torch.float32)).numpy().ravel()*rs+rm)
        pred=np.maximum(pred,0.0)
        jdump(dict(model=model,seed=s,recomputed=True,
            rmse=float(np.sqrt(np.mean((pred-rte)**2))),
            mae=float(np.mean(np.abs(pred-rte))),
            mean_rul_test=float(np.mean(rte)),sd_rul_test=float(np.std(rte))),fp)
        print("allmodes",model,s,"ok")
print("PART1 DONE")


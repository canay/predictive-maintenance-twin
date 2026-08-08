import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from simulator import fit_params, fit_pools, gen_pool, pool_to_arrays
T0=time.time(); MAX_SEC=30
OUT=BASE+"/results/rul"; os.makedirs(OUT,exist_ok=True)
DD=BASE+"/data"
df=load_ai4i(); P=fit_params(df); POOLS=fit_pools(df)
ALPHA=0.3; FLOOR=3.0
def get_pool(tag,seed,n):
    fp=f"{DD}/pool2_{tag}_{seed}.npz"
    if os.path.exists(fp):
        z=np.load(fp,allow_pickle=True); return z["X"],z["rul"],z["eid"],list(z["modes"])
    eps=gen_pool({"tr":1000,"te":2000,"va":3000,"dtr":5000,"dte":6000}[tag]+seed,n,P,POOLS,shocks=not tag.startswith("d"))
    X,rul,eid=pool_to_arrays(eps)
    modes=[e["fail_mode"] for e in eps]
    np.savez_compressed(fp,X=X,rul=rul,eid=eid,modes=np.array(modes))
    return X,rul,eid,modes
def alpha_lambda(rul,pred,eid,modes):
    accs={}; horizons=[]; lam_grid=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    ok=np.abs(pred-rul)<=np.maximum(ALPHA*rul,FLOOR)
    for lam in lam_grid:
        hits=[]
        for e in np.unique(eid):
            m=eid==e; T=m.sum()
            t=int(round(lam*(T-1)))
            hits.append(bool(ok[m][t]))
        accs[str(lam)]=float(np.mean(hits))
    hor2=[]
    for e in np.unique(eid):
        m=eid==e; T=m.sum(); o=ok[m]
        o2=np.abs(pred[m]-rul[m])<=0.2*T
        t0=T
        for t in range(T-1,-1,-1):
            if o[t]: t0=t
            else: break
        horizons.append((T-t0)/T)
        t1=T
        for t in range(T-1,-1,-1):
            if o2[t]: t1=t
            else: break
        hor2.append((T-t1)/T)
    return accs,float(np.mean(horizons)),float(np.mean(hor2))
def run_unit(model,seed):
    Xtr,rtr,_,_=get_pool("dtr",seed,400)
    Xte,rte,eid,modes=get_pool("dte",seed,200)
    if model=="HGBR":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.08,random_state=seed)
        m.fit(Xtr,rtr); pred=m.predict(Xte)
        import joblib; joblib.dump(m,f"{DD}/rulmodel2_HGBR_{seed}.joblib")
    else:
        import torch, torch.nn as nn
        torch.manual_seed(seed); torch.set_num_threads(2)
        mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9; rm,rs=rtr.mean(),rtr.std()
        Xt=torch.tensor((Xtr-mu)/sd,dtype=torch.float32); yt=torch.tensor((rtr-rm)/rs,dtype=torch.float32)[:,None]
        net=nn.Sequential(nn.Linear(Xtr.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        opt=torch.optim.Adam(net.parameters(),lr=1e-3); lo=nn.MSELoss()
        n=len(Xt); idx=np.arange(n)
        for ep in range(60):
            np.random.default_rng(seed*100+ep).shuffle(idx)
            for i in range(0,n,256):
                b=idx[i:i+256]; opt.zero_grad(); l=lo(net(Xt[b]),yt[b]); l.backward(); opt.step()
        with torch.no_grad():
            pred=(net(torch.tensor((Xte-mu)/sd,dtype=torch.float32)).numpy().ravel()*rs+rm)
    pred=np.maximum(pred,0.0)
    wearmask=np.isin(np.array(modes),["TWF","OSF"])[eid.astype(int)]
    res=dict(model=model,seed=seed,
        rmse=float(np.sqrt(np.mean((pred-rte)**2))),mae=float(np.mean(np.abs(pred-rte))),
        rmse_wear=float(np.sqrt(np.mean((pred[wearmask]-rte[wearmask])**2))),
        mae_wear=float(np.mean(np.abs(pred[wearmask]-rte[wearmask]))))
    accs,hor,hor2=alpha_lambda(rte,pred,eid,np.array(modes))
    res["alpha_lambda"]=accs; res["prog_horizon_frac"]=hor; res["ph02_frac"]=hor2
    if seed==0:
        np.savez_compressed(f"{OUT}/preds_{model}_s0.npz",pred=pred,rul=rte,eid=eid,modes=np.array(modes))
    return res
units=[(m,s) for m in ["HGBR","MLP"] for s in range(5)]
for model,seed in units:
    fp=f"{OUT}/u2_{model}_{seed}.json"
    if os.path.exists(fp): continue
    if time.time()-T0>MAX_SEC: print("BUDGET"); sys.exit(2)
    res=run_unit(model,seed); jdump(res,fp); print(model,seed,round(res["rmse"],2),round(res["mae"],2))
print("ALL DONE")


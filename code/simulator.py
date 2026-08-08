import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
C=2*np.pi/60.0
OSF_THR={0:11000.0,1:12000.0,2:13000.0}  # L,M,H

def fit_params(df):
    air=df["Air temperature [K]"].values
    proc=df["Process temperature [K]"].values
    dT=proc-air
    x=air-air.mean(); rho=float(np.corrcoef(x[:-1],x[1:])[0,1])
    sig_eps=float(np.std(x)*np.sqrt(1-rho**2))
    d=np.diff(df["Tool wear [min]"].values)
    inc=d[(d>0)&(d<10)]
    tw_f=df["Tool wear [min]"].values[df["TWF"].values==1]
    p=dict(air_mu=float(air.mean()), air_rho=rho, air_sig=sig_eps,
        air_sd_stat=float(np.std(air)),
        twf_lo=float(tw_f.min()), twf_hi=float(tw_f.max()),
        inc_vals=[float(v) for v in np.unique(inc)],
        inc_probs=[float((inc==v).mean()) for v in np.unique(inc)],
        type_probs=[float((df["Type"]==t).mean()) for t in ["L","M","H"]])
    s=sum(p["inc_probs"]); p["inc_probs"]=[q/s for q in p["inc_probs"]]
    return p

def fit_pools(df):
    air=df["Air temperature [K]"].values; proc=df["Process temperature [K]"].values
    return dict(dT=proc-air, tq=df["Torque [Nm]"].values, rpm=df["Rotational speed [rpm]"].values)

def gen_episode(rng,p,pools,max_cycles=400,shocks=True):
    ty=int(rng.choice(3,p=p["type_probs"]))
    w_twf=rng.uniform(p["twf_lo"],p["twf_hi"])
    air=p["air_mu"]+rng.normal(0,p["air_sd_stat"])
    n=len(pools["tq"])
    wear=0.0; rows=[]; fail_mode=None
    for t in range(max_cycles):
        air=p["air_mu"]+p["air_rho"]*(air-p["air_mu"])+rng.normal(0,p["air_sig"])
        j=int(rng.integers(n))
        dT=float(pools["dT"][j]); tq=float(pools["tq"][j]); rpm=float(pools["rpm"][j])
        proc=air+dT; pw=tq*rpm*C
        wear+=float(rng.choice(p["inc_vals"],p=p["inc_probs"]))
        rows.append((ty,air,proc,rpm,tq,wear))
        if wear>=w_twf: fail_mode="TWF"
        elif wear*tq>OSF_THR[ty]: fail_mode="OSF"
        elif shocks and dT<8.6 and rpm<1380: fail_mode="HDF"
        elif shocks and (pw<3500 or pw>9000): fail_mode="PWF"
        elif shocks and rng.random()<0.001: fail_mode="RNF"
        if fail_mode: break
    X=np.array(rows); T=len(rows)
    rul=np.arange(T-1,-1,-1,dtype=float)
    return dict(X=X,rul=rul,fail_mode=fail_mode or "CENSOR",T=T,w_twf=float(w_twf))

def gen_pool(seed,n_ep,p,pools,shocks=True):
    rng=np.random.default_rng(seed)
    return [gen_episode(rng,p,pools,shocks=shocks) for _ in range(n_ep)]

def pool_to_arrays(eps):
    X=np.vstack([e["X"] for e in eps])
    rul=np.concatenate([e["rul"] for e in eps])
    eid=np.concatenate([np.full(e["T"],i) for i,e in enumerate(eps)])
    return X,rul,eid


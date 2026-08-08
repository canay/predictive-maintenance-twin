import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
DD=BASE+"/data"; OUT=BASE+"/results"
EVAL_SEEDS=list(range(15))
MODEL_SEEDS=list(range(5))
SP=json.load(open(OUT+"/sim_params.json"))
MEAN_INC=float(np.dot(SP["inc_vals"],SP["inc_probs"]))
CP,CD_RATE,DP,DC=1.0,0.05,2,10
TAUS=[0.5,0.6,0.7,0.8,0.9,0.95]; HS=[2,3,5,8,12,16]; KS=[20,30,40,50,60,80]
RATIOS=[2,5,10,20]
def load(tag,seed):
    z=np.load(f"{DD}/pool2_{tag}_{seed}.npz",allow_pickle=True)
    g=np.load(f"{OUT}/gate_{tag}_{seed}.npz",allow_pickle=True)
    gate_full=np.zeros(len(z["X"]),bool); gate_full[g["cand"]]=g["gate"]
    eid=z["eid"]; bounds=np.searchsorted(eid,np.arange(eid.max()+2))
    eps=[]
    for e in range(int(eid.max())+1):
        a,b=bounds[e],bounds[e+1]
        eps.append(dict(T=b-a,risk=g["risk"][a:b],rul=g["rul"][a:b],gate=gate_full[a:b],
            wear=z["X"][a:b,5],wtwf=float(z["wtwf"][e]),mode=str(z["modes"][e])))
    return eps
def ep_outcome(ep,policy,param):
    T=ep["T"]
    if policy=="reactive": t=None
    elif policy=="periodic": t=param-1 if param-1<=T-2 else None
    else:
        if policy=="risk": trig=ep["risk"]>=param
        elif policy=="rul": trig=ep["rul"]<=param
        elif policy=="rul_gated": trig=(ep["rul"]<=param)&ep["gate"]
        elif policy=="risk_gated": trig=(ep["risk"]>=param)&ep["gate"]
        w=np.where(trig[:T-1])[0]
        t=int(w[0]) if len(w) else None
    if t is None:
        return dict(prev=0,fail=1,op=T,down=DC,mode=ep["mode"],false_act=0)
    rem=(ep["wtwf"]-ep["wear"][t])/MEAN_INC
    return dict(prev=1,fail=0,op=t+1,down=DP,mode=None,false_act=int(rem>15),rem=rem,false30=int(rem>30))
def evaluate(eps,policy,param,ratio):
    os_=[ep_outcome(ep,policy,param) for ep in eps]
    op=sum(o["op"] for o in os_); down=sum(o["down"] for o in os_)
    np_=sum(o["prev"] for o in os_); nf=sum(o["fail"] for o in os_)
    cost=np_*CP+nf*ratio*CP+CD_RATE*down
    wearfail=sum(1 for o in os_ if o["mode"] in ("TWF","OSF"))
    fa=sum(o["false_act"] for o in os_)
    f30=sum(o.get("false30",0) for o in os_)
    rems=[o["rem"] for o in os_ if "rem" in o]
    return dict(cost_rate=cost/op,avail=op/(op+down),fails_per_1000=1000*nf/op,
        n_prev=np_,n_fail=nf,wear_fails=wearfail,false_actions=fa,
        false_action_rate=fa/np_ if np_>0 else 0.0,
        false30=f30,false30_rate=f30/np_ if np_>0 else 0.0,
        mean_rem_at_prev=float(np.mean(rems)) if rems else 0.0,op=op)
pv=load("pv",0)
sel={}
for ratio in RATIOS:
    best={}
    for pol,grid in [("periodic",KS),("risk",TAUS),("rul",HS)]:
        cs=[(evaluate(pv,pol,g,ratio)["cost_rate"],g) for g in grid]
        best[pol]=min(cs)[1]
    sel[str(ratio)]=best
res=dict(selection=sel,mean_inc=MEAN_INC,cost_model=dict(cp=CP,cd_rate=CD_RATE,dp=DP,dc=DC),
    grids=dict(taus=TAUS,hs=HS,Ks=KS),per_ratio={},eval_seeds=EVAL_SEEDS,
    model_seed_mapping={str(s): int(s % len(MODEL_SEEDS)) for s in EVAL_SEEDS},
    design_note="Policy evaluation uses 15 independent simulator evaluation pools; saved risk/RUL model seeds 0-4 are reused cyclically to avoid a new model-training experiment.")
pools={s: load("pe",s) for s in EVAL_SEEDS}
for ratio in RATIOS:
    b=sel[str(ratio)]
    pols=[("reactive",None),("periodic",b["periodic"]),("risk",b["risk"]),
          ("rul",b["rul"]),("risk_gated",b["risk"]),("rul_gated",b["rul"])]
    out={}
    for pol,param in pols:
        per_seed=[evaluate(pools[s],pol,param,ratio) for s in EVAL_SEEDS]
        out[pol]=dict(param=param,
            per_seed={k:[p[k] for p in per_seed] for k in per_seed[0]},
            mean={k:float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]},
            sd={k:float(np.std([p[k] for p in per_seed])) for k in per_seed[0]})
    res["per_ratio"][str(ratio)]=out
# paired tests across (seed,ratio) on cost_rate; retained as a consistency summary.
from scipy.stats import wilcoxon
def pairs(a,b):
    x=[];y=[]
    for r in RATIOS:
        x+=res["per_ratio"][str(r)][a]["per_seed"]["cost_rate"]
        y+=res["per_ratio"][str(r)][b]["per_seed"]["cost_rate"]
    return np.array(x),np.array(y)
tests={}
for a,b in [("risk","rul"),("rul","rul_gated"),("risk","risk_gated"),("periodic","rul"),("reactive","rul")]:
    x,y=pairs(a,b)
    try: w=wilcoxon(x,y); tests[f"{a}_vs_{b}"]=dict(stat=float(w.statistic),p=float(w.pvalue),mean_delta=float(np.mean(x-y)))
    except Exception as ex: tests[f"{a}_vs_{b}"]=dict(error=str(ex))
res["tests"]=tests
per_ratio_tests={}
for ratio in RATIOS:
    rkey=str(ratio)
    per_ratio_tests[rkey]={}
    for a,b in [("risk","rul"),("rul","rul_gated"),("risk","risk_gated"),("periodic","rul"),("reactive","rul")]:
        x=np.array(res["per_ratio"][rkey][a]["per_seed"]["cost_rate"])
        y=np.array(res["per_ratio"][rkey][b]["per_seed"]["cost_rate"])
        try:
            w=wilcoxon(x,y) if np.any(x-y) else None
            per_ratio_tests[rkey][f"{a}_vs_{b}"]=dict(
                stat=float(w.statistic) if w is not None else 0.0,
                p=float(w.pvalue) if w is not None else 1.0,
                mean_delta=float(np.mean(x-y)))
        except Exception as ex:
            per_ratio_tests[rkey][f"{a}_vs_{b}"]=dict(error=str(ex))
res["per_ratio_tests"]=per_ratio_tests
jdump(res,OUT+"/policy_eval.json")
print("selection",sel)
for ratio in ["5","20"]:
    print("ratio",ratio,{p:round(res["per_ratio"][ratio][p]["mean"]["cost_rate"],4) for p in res["per_ratio"][ratio]})
print({k:(round(v.get("p",9),5),round(v.get("mean_delta",0),4)) for k,v in tests.items()})


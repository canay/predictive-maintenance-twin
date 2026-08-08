import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import wilcoxon
OUT=BASE+"/results"
S=json.load(open(OUT+"/results_summary.json"))
P=S["policy"]; RATIOS=["2","5","10","20"]
EVAL_SEEDS=list(P.get("eval_seeds", range(len(P["per_ratio"]["2"]["reactive"]["per_seed"]["cost_rate"]))))
ex={}
# all-modes RUL aggregate
am={}
for m in ["HGBR","MLP"]:
    rs=[json.load(open(f)) for f in sorted(glob.glob(OUT+"/rul_allmodes/u_%s_*.json"%m))]
    am[m]=dict(rmse=dict(mean=float(np.mean([r["rmse"] for r in rs])),sd=float(np.std([r["rmse"] for r in rs]))),
               mae=dict(mean=float(np.mean([r["mae"] for r in rs])),sd=float(np.std([r["mae"] for r in rs]))),
               mean_rul_test=float(np.mean([r["mean_rul_test"] for r in rs])))
ex["rul_allmodes"]=am
# tool-life RUL HGBR vs MLP paired Wilcoxon (5 seeds)
h=[json.load(open(f))["rmse"] for f in sorted(glob.glob(OUT+"/rul/u2_HGBR_*.json"))]
ml=[json.load(open(f))["rmse"] for f in sorted(glob.glob(OUT+"/rul/u2_MLP_*.json"))]
w=wilcoxon(h,ml)
ex["rul_hgbr_vs_mlp_rmse"]=dict(hgbr=h,mlp=ml,stat=float(w.statistic),p=float(w.pvalue),
    mean_delta=float(np.mean(np.array(h)-np.array(ml))))
# per-ratio savings vs reactive (percent, per-seed) + Wilcoxon
sav={}
for r in RATIOS:
    base=np.array(P["per_ratio"][r]["reactive"]["per_seed"]["cost_rate"])
    sav[r]={}
    for pol in ["periodic","risk","rul","risk_gated","rul_gated"]:
        c=np.array(P["per_ratio"][r][pol]["per_seed"]["cost_rate"])
        pct=100*(base-c)/base
        try:
            ww=wilcoxon(base,c); st,pv=float(ww.statistic),float(ww.pvalue)
        except Exception:
            st,pv=None,None
        sav[r][pol]=dict(pct_mean=float(pct.mean()),pct_sd=float(pct.std()),
            per_seed_pct=[float(v) for v in pct],wilcoxon_stat=st,wilcoxon_p=pv)
ex["savings_vs_reactive_pct"]=sav
# prognostic-layer delta (risk -> rul), percent per ratio
pr={}
for r in RATIOS:
    a=np.array(P["per_ratio"][r]["risk"]["per_seed"]["cost_rate"])
    b=np.array(P["per_ratio"][r]["rul"]["per_seed"]["cost_rate"])
    pct=100*(a-b)/a
    ww=wilcoxon(a,b)
    pr[r]=dict(pct_mean=float(pct.mean()),pct_sd=float(pct.std()),wilcoxon_stat=float(ww.statistic),wilcoxon_p=float(ww.pvalue))
ex["prognostic_layer_cost_reduction_pct"]=pr
# gate effect: cost increase pct and false30 reduction; per-ratio Wilcoxon plus a pooled consistency summary.
gc={}
gc_tests={}
xa,xb,fa,fb,na,nb=[],[],[],[],[],[]
for r in RATIOS:
    a=np.array(P["per_ratio"][r]["rul"]["per_seed"]["cost_rate"])
    b=np.array(P["per_ratio"][r]["rul_gated"]["per_seed"]["cost_rate"])
    f1=np.array(P["per_ratio"][r]["rul"]["per_seed"]["false30_rate"])
    f2=np.array(P["per_ratio"][r]["rul_gated"]["per_seed"]["false30_rate"])
    pct=100*(b-a)/a
    try:
        ww_cost=wilcoxon(b,a)
        cost_p=float(ww_cost.pvalue)
        cost_stat=float(ww_cost.statistic)
    except Exception:
        cost_p,cost_stat=None,None
    try:
        ww_false=wilcoxon(f1,f2)
        false_p=float(ww_false.pvalue)
        false_stat=float(ww_false.statistic)
    except Exception:
        false_p,false_stat=None,None
    gc[r]=dict(cost_increase_pct_mean=float(pct.mean()),cost_increase_pct_sd=float(pct.std()),
        false30_ungated=float(f1.mean()),false30_gated=float(f2.mean()))
    gc_tests[r]=dict(cost_wilcoxon_stat=cost_stat,cost_wilcoxon_p=cost_p,
        false30_wilcoxon_stat=false_stat,false30_wilcoxon_p=false_p,
        false30_mean_delta=float(np.mean(f1-f2)))
    xa+=list(a); xb+=list(b); fa+=list(f1); fb+=list(f2)
    na+=list(np.array(P["per_ratio"][r]["rul"]["per_seed"]["n_prev"]))
    nb+=list(np.array(P["per_ratio"][r]["rul_gated"]["per_seed"]["n_prev"]))
wf=wilcoxon(fa,fb)
ex["gate_effect"]=dict(per_ratio=gc,
    per_ratio_tests=gc_tests,
    false30_wilcoxon=dict(stat=float(wf.statistic),p=float(wf.pvalue),
        mean_ungated=float(np.mean(fa)),mean_gated=float(np.mean(fb))),
    n_prev_mean_ungated=float(np.mean(na)),n_prev_mean_gated=float(np.mean(nb)))
# gate pass rates per seed
gp={}
for s in EVAL_SEEDS:
    z=np.load(OUT+f"/gate_pe_{s}.npz")
    gp[str(s)]=dict(n_cand=int(len(z["cand"])),pass_rate=float(z["gate"].mean()))
ex["gate_pass_rates_pe"]=gp
ex["gate_pass_rate_mean"]=float(np.mean([v["pass_rate"] for v in gp.values()]))
# degradation pool stats: mean tool life
Ls=[]
for s in range(5):
    z=np.load(BASE+f"/data/pool2_dte_{s}.npz",allow_pickle=True)
    Ls+=list(np.bincount(z["eid"].astype(int)))
ex["deg_test_episode_len"]=dict(mean=float(np.mean(Ls)),sd=float(np.std(Ls)),n=len(Ls))
# policy eval pool stats
Lp=[]
for s in EVAL_SEEDS:
    z=np.load(BASE+f"/data/pool2_pe_{s}.npz",allow_pickle=True)
    Lp+=list(z["Ts"])
ex["policy_eval_episode_len"]=dict(mean=float(np.mean(Lp)),sd=float(np.std(Lp)),n=len(Lp))
# risk RF vs HGB AUROC paired Wilcoxon per target (n=5)
rb={}
for t in ["Machine failure","TWF","HDF","PWF","OSF"]:
    a=[json.load(open(OUT+"/risk/u_RF_%s_%d.json"%(t.replace(' ','_'),s)))["auroc"] for s in range(5)]
    b=[json.load(open(OUT+"/risk/u_HGB_%s_%d.json"%(t.replace(' ','_'),s)))["auroc"] for s in range(5)]
    try:
        ww=wilcoxon(a,b); st,pv=float(ww.statistic),float(ww.pvalue)
    except Exception:
        st,pv=None,None
    rb[t]=dict(rf_mean=float(np.mean(a)),hgb_mean=float(np.mean(b)),stat=st,p=pv)
ex["risk_rf_vs_hgb_auroc"]=rb
jdump(ex,OUT+"/stats_extra.json")
print(json.dumps({k:(v if not isinstance(v,dict) else "...") for k,v in ex.items()},indent=1))
print("rul test wilcoxon p",ex["rul_hgbr_vs_mlp_rmse"]["p"])
print("gate false30 p",ex["gate_effect"]["false30_wilcoxon"])
print("savings r20 rul",ex["savings_vs_reactive_pct"]["20"]["rul"])
print("deg ep len",ex["deg_test_episode_len"])


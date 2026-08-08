import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from simulator import fit_params, fit_pools, gen_pool, pool_to_arrays, C
from scipy import stats
df=load_ai4i()
p=fit_params(df); pools=fit_pools(df)
jdump(p,BASE+"/results/sim_params.json")
eps=gen_pool(12345,600,p,pools)
X,rul,eid=pool_to_arrays(eps)
sim=dict(air=X[:,1],proc=X[:,2],rpm=X[:,3],tq=X[:,4],wear=X[:,5])
real=dict(air=df["Air temperature [K]"].values,proc=df["Process temperature [K]"].values,
 rpm=df["Rotational speed [rpm]"].values,tq=df["Torque [Nm]"].values,wear=df["Tool wear [min]"].values)
calib={}
for k in sim:
    ks=stats.ks_2samp(real[k],sim[k]); w=stats.wasserstein_distance(real[k],sim[k])
    calib[k]=dict(ks=float(ks.statistic),wass=float(w),
        real_mu=float(np.mean(real[k])),sim_mu=float(np.mean(sim[k])),
        real_sd=float(np.std(real[k])),sim_sd=float(np.std(sim[k])))
n_cyc=X.shape[0]; modes=["TWF","HDF","PWF","OSF","RNF"]
fm={m:sum(1 for e in eps if e["fail_mode"]==m) for m in modes}
calib["failure_rates_per_1000_cycles"]={m:1000.0*fm[m]/n_cyc for m in modes}
calib["failure_rates_real_per_1000_rows"]={m:1000.0*df[m].sum()/len(df) for m in modes}
calib["any_failure_rate_sim"]=1000.0*sum(fm.values())/n_cyc
calib["any_failure_rate_real"]=1000.0*float((df[modes].sum(axis=1)>0).sum())/len(df)
calib["n_sim_cycles"]=int(n_cyc); calib["n_episodes"]=len(eps)
calib["mode_counts_sim"]=fm
calib["mode_share_sim"]={m:fm[m]/max(sum(fm.values()),1) for m in modes}
calib["mode_share_real"]={m:float(df[m].sum())/float(sum(df[mm].sum() for mm in modes)) for m in modes}
calib["mean_episode_len"]=float(np.mean([e["T"] for e in eps]))
calib["censor_frac"]=float(np.mean([e["fail_mode"]=="CENSOR" for e in eps]))
tq=real["tq"]; rpm=real["rpm"]; pw=tq*rpm*C
hdf=((df["Process temperature [K]"]-df["Air temperature [K]"])<8.6)&(rpm<1380)
thr=df["Type"].map({"L":11000,"M":12000,"H":13000}).values
osf=real["wear"]*tq>thr
pwf=(pw<3500)|(pw>9000)
calib["rule_check"]={"HDF":[int(hdf.sum()),int(df.HDF.sum()),int((hdf&(df.HDF==1)).sum())],
 "OSF":[int(osf.sum()),int(df.OSF.sum()),int((osf&(df.OSF==1)).sum())],
 "PWF":[int(pwf.sum()),int(df.PWF.sum()),int((pwf&(df.PWF==1)).sum())]}
jdump(calib,BASE+"/results/calibration.json")
print({k:round(calib[k]["ks"],4) for k in sim})
print("rates sim",{m:round(calib['failure_rates_per_1000_cycles'][m],2) for m in modes})
print("rates real",calib["failure_rates_real_per_1000_rows"])
print("share sim",{m:round(calib['mode_share_sim'][m],3) for m in modes})
print("share real",{m:round(calib['mode_share_real'][m],3) for m in modes})
print("mean_len",calib["mean_episode_len"],"censor",calib["censor_frac"])


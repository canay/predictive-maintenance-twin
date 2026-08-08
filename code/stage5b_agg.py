import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import spearmanr
from itertools import combinations
GT = {"TWF": [5], "HDF": [1, 2, 3], "PWF": [3, 4], "OSF": [5, 4, 0], "RNF": []}
FE = ["Type", "Air T", "Process T", "Rot. speed", "Torque", "Tool wear"]
out = {}
for t in ["TWF", "HDF", "PWF", "OSF", "RNF", "Machine_failure"]:
    rs = [json.load(open(f)) for f in sorted(glob.glob(BASE + f"/results/shap/u_{t}_*.json"))]
    imps_glob = [np.array(r["imp"]) for r in rs]
    ps = [json.load(open(f)) for f in sorted(glob.glob(BASE + f"/results/shap/p_{t}_*.json"))]
    imps_pos = [np.array(r["imp_posall"]) for r in ps] if ps else imps_glob
    gt = GT.get(t, [])
    k = len(gt)
    def hits_of(imps):
        return [len(set(np.argsort(-im)[:k]) & set(gt)) / k for im in imps] if k > 0 else None
    def stab(imps):
        j3, sp = [], []
        for a, b in combinations(range(len(imps)), 2):
            ta, tb = set(np.argsort(-imps[a])[:3]), set(np.argsort(-imps[b])[:3])
            j3.append(len(ta & tb) / len(ta | tb))
            sp.append(spearmanr(imps[a], imps[b]).statistic)
        return (float(np.mean(j3)), float(np.std(j3)), float(np.mean(sp)), float(np.std(sp)))
    hp = hits_of(imps_pos); hg = hits_of(imps_glob)
    j3m, j3s, spm, sps = stab(imps_pos)
    j3mg, j3sg, spmg, spsg = stab(imps_glob)
    out[t] = dict(
        hit_rates=hp, hit_mean=float(np.mean(hp)) if hp else None,
        hit_rates_global=hg, hit_mean_global=float(np.mean(hg)) if hg else None,
        jaccard_top3=j3m, jaccard_sd=j3s, spearman=spm, spearman_sd=sps,
        jaccard_top3_global=j3mg, spearman_global=spmg,
        mean_imp_pos=np.mean(imps_pos, 0).tolist(), sd_imp_pos=np.std(imps_pos, 0).tolist(),
        mean_imp_global=np.mean(imps_glob, 0).tolist(),
        top3=[FE[i] for i in np.argsort(-np.mean(imps_pos, 0))[:3]],
        gt=[FE[i] for i in gt], n_pos=ps[0]["n"] if ps else None)
jdump(out, BASE + "/results/shap_summary.json")
for t, d in out.items():
    print(t, "hit", d["hit_mean"], "hitG", d["hit_mean_global"],
          "J3 %.2f" % d["jaccard_top3"], "rho %.3f" % d["spearman"],
          "top3", d["top3"], "gt", d["gt"])


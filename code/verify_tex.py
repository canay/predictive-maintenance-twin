import json, os, re, sys, glob
import numpy as np
B=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S=json.load(open(B+"/results/results_summary.json"))
EX=json.load(open(B+"/results/stats_extra.json"))
SP=json.load(open(B+"/results/sim_params.json"))
Q1=json.load(open(B+"/results/q1_sensitivity.json"))
Q1R=json.load(open(B+"/results/q1_robustness.json"))
HW=json.load(open(B+"/results/q1_high_wear_sensitivity.json"))
MSA=json.load(open(B+"/results/q1_model_seed_audit.json"))
tex=open(os.path.join(B, "manuscript", "SCI-f08-predictive_maintenance_twin.tex"), encoding="utf-8").read()
fails=[]
def ck(name,cond):
    if not cond: fails.append(name)
    print(("OK " if cond else "FAIL ")+name)
def near(a,b,tol): return abs(a-b)<=tol
def intex(s): 
    return s in tex
# dataset prevalences
import pandas as pd
df=pd.read_csv(B+"/data/ai4i2020.csv")
ck("prev339", df["Machine failure"].sum()==339 and "339 positives" in tex)
for m,n in [("TWF",46),("HDF",115),("PWF",95),("OSF",98),("RNF",19)]:
    ck("prev_"+m, df[m].sum()==n and ("(%s, %d)"%(m,n)) in tex)
# sim params
ck("rho", near(SP["air_rho"],0.99942,5e-6) and "0.99942" in tex)
ck("sig", near(SP["air_sig"],0.0683,5e-5) and "0.0683" in tex)
ck("mu_a", near(SP["air_mu"],300.005,5e-4) and "300.005" in tex)
ck("twf_win", SP["twf_lo"]==198 and SP["twf_hi"]==253 and "\\mathcal{U}(198,253)" in tex)
ck("type_probs", all(near(a,b,5e-4) for a,b in zip(SP["type_probs"],[0.600,0.300,0.100])) and "(0.600, 0.300, 0.100)" in tex)
ck("mean_inc", near(S["mean_wear_increment"],2.60,0.005) and "2.60" in tex)
# rule check
rc=S["calibration"]["rule_check"]
ck("rules_exact", rc["HDF"]==[115,115,115] and rc["OSF"]==[98,98,98] and rc["PWF"]==[95,95,95] and "115/115 HDF, 98/98 OSF, and 95/95 PWF" in tex)
# risk table rows: regenerate strings and check presence
def ms(d,f=3): return "%.*f $\\pm$ %.*f"%(f,d["mean"],f,d["sd"])
for t in ["Machine failure","TWF","HDF","PWF","OSF","RNF"]:
    for m in ["LR","RF","HGB"]:
        d=S["risk"][m][t]
        row="%s & %s & %s & %s & %s"%(m,ms(d["auroc"]),ms(d["ap"]),ms(d["f1"]),ms(d["ece"]))
        ck("riskrow_%s_%s"%(m,t.replace(' ','_')), row in tex)
# risk text claims
r=S["risk"]
ck("rf_mf", near(r["RF"]["Machine failure"]["auroc"]["mean"],0.972,5e-4) and near(r["RF"]["Machine failure"]["ap"]["mean"],0.762,5e-4) and near(r["RF"]["Machine failure"]["ece"]["mean"],0.013,5e-4))
ck("hgb_hdf", near(r["HGB"]["HDF"]["auroc"]["mean"],0.9995,5e-5))
ck("hgb_pwf", near(r["HGB"]["PWF"]["auroc"]["mean"],0.998,5e-4))
ck("lr_osf", near(r["LR"]["OSF"]["auroc"]["mean"],0.9996,5e-5))
ck("lr_twf", near(r["LR"]["TWF"]["auroc"]["mean"],0.968,5e-4))
ck("twf_ap_lt_015", max(r[m]["TWF"]["ap"]["mean"] for m in r)<0.15)
ck("n_pos_twf", r["LR"]["TWF"]["n_pos_test"]==14)
ck("rnf_chance", all(0.4<r[m]["RNF"]["auroc"]["mean"]<0.6 for m in r))
ck("rf_vs_hgb_p", min(v["p"] for v in EX["risk_rf_vs_hgb_auroc"].values())>=0.0625)
# calibration
C=S["calibration"]
ck("cal_pool", C["n_episodes"]==600 and C["n_sim_cycles"]==23145 and near(C["mean_episode_len"],38.6,0.05) and C["censor_frac"]==0.0)
for k,v in [("tq",0.004),("rpm",0.004),("air",0.069),("proc",0.113),("wear",0.216)]:
    ck("ks_"+k, near(C[k]["ks"],v,5e-4))
fr=C["failure_rates_per_1000_cycles"]; fre=C["failure_rates_real_per_1000_rows"]
for m,s_,r_ in [("HDF",11.7,11.5),("PWF",8.5,9.5),("RNF",1.0,1.9),("TWF",2.1,4.6),("OSF",2.6,9.8)]:
    ck("rate_"+m, near(fr[m],s_,0.05) and near(fre[m],r_,0.05))
ck("wear_mu", near(C["wear"]["sim_mu"],76.65,0.05) and near(C["wear"]["real_mu"],107.95,0.005))
# calib table rows
nm={"air":"Air temperature [K]","proc":"Process temperature [K]","rpm":"Rotational speed [rpm]","tq":"Torque [Nm]","wear":"Tool wear [min]"}
for k in nm:
    c=C[k]
    row="%s & %.2f / %.2f & %.2f / %.2f & %.3f & %.3f"%(nm[k],c["real_mu"],c["real_sd"],c["sim_mu"],c["sim_sd"],c["ks"],c["wass"])
    ck("calrow_"+k, row in tex)
# RUL
R=S["rul"]
for m in ["HGBR","MLP"]:
    d=R[m]
    row="%s & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f"%(m,
        d["rmse"]["mean"],d["rmse"]["sd"],d["mae"]["mean"],d["mae"]["sd"],
        d["alpha_lambda"]["0.3"]["mean"],d["alpha_lambda"]["0.3"]["sd"],
        d["alpha_lambda"]["0.5"]["mean"],d["alpha_lambda"]["0.5"]["sd"],
        d["alpha_lambda"]["0.9"]["mean"],d["alpha_lambda"]["0.9"]["sd"],
        d["ph02"]["mean"],d["ph02"]["sd"])
    ck("rulrow_"+m, row in tex)
ck("rul_wilcoxon", near(EX["rul_hgbr_vs_mlp_rmse"]["p"],0.44,0.005))
ck("deg_eplen", near(EX["deg_test_episode_len"]["mean"],81.4,0.05) and near(EX["deg_test_episode_len"]["sd"],7.0,0.05))
am=EX["rul_allmodes"]
ck("allmodes", near(am["HGBR"]["rmse"]["mean"],20.86,0.005) and near(am["HGBR"]["rmse"]["sd"],0.53,0.005) and near(am["MLP"]["rmse"]["mean"],20.59,0.005) and near(am["MLP"]["rmse"]["sd"],0.48,0.005) and near(am["HGBR"]["mean_rul_test"],29.5,0.05))
ck("allmodes_intex", "20.86\\pm0.53" in tex and "20.59\\pm0.48" in tex and "29.5 cycles" in tex)
# policy table rows
P=S["policy"]
names={"reactive":"Reactive","periodic":"Periodic","risk":"Risk-triggered","rul":"RUL-aware","risk_gated":"Risk-triggered, gated","rul_gated":"RUL-aware, gated"}
for rt in ["2","5","10","20"]:
    for p in ["reactive","periodic","risk","risk_gated","rul","rul_gated"]:
        d=P["per_ratio"][rt][p]; m=d["mean"]; sd=d["sd"]
        if p=="reactive": pa="--"
        elif p=="periodic": pa="$K{=}%d$"%d["param"]
        elif p.startswith("risk"): pa="$\\tau{=}%.1f$"%d["param"]
        else: pa="$h{=}%d$"%d["param"]
        if p=="reactive": sv="--"
        else:
            v=EX["savings_vs_reactive_pct"][rt][p]["pct_mean"]
            sv=("$-%.1f$"%abs(v)) if round(v,1)<0 else "%.1f"%v
        row="%s & %s & %.4f $\\pm$ %.4f & %s & %.3f & %.1f & %.1f & %.2f"%(names[p],pa,m["cost_rate"],sd["cost_rate"],sv,m["avail"],m["fails_per_1000"],m["n_prev"],m["false30_rate"])
        ck("polrow_r%s_%s"%(rt,p), row in tex)
# policy text claims
T=P["tests"]
ck("pooled_risk_rul", near(T["risk_vs_rul"]["p"],1.63e-11,1e-12))
ck("pooled_reactive_rul", near(T["reactive_vs_rul"]["p"],1.63e-11,1e-12))
ck("pooled_periodic", near(T["periodic_vs_rul"]["p"],1.64e-4,1e-5))
ck("pooled_gate_cost", near(T["rul_vs_rul_gated"]["p"],3.29e-11,2e-12))
ck("pooled_riskgate", near(T["risk_vs_risk_gated"]["p"],2.07e-5,1e-6))
ck("gate_false30", near(EX["gate_effect"]["false30_wilcoxon"]["p"],6.69e-9,1e-10))
sv=EX["savings_vs_reactive_pct"]
ck("sav_seq", near(sv["2"]["rul"]["pct_mean"],6.4,0.05) and near(sv["5"]["rul"]["pct_mean"],11.3,0.05) and near(sv["10"]["rul"]["pct_mean"],13.7,0.05) and near(sv["20"]["rul"]["pct_mean"],15.0,0.05))
ck("sav_sd", near(sv["2"]["rul"]["pct_sd"],0.9,0.05) and near(sv["5"]["rul"]["pct_sd"],1.5,0.05) and near(sv["10"]["rul"]["pct_sd"],1.8,0.05) and near(sv["20"]["rul"]["pct_sd"],1.9,0.05))
ck("risk_best_sav", near(max(sv[rt]["risk"]["pct_mean"] for rt in sv),1.6,0.05))
pr=EX["prognostic_layer_cost_reduction_pct"]
ck("prog_range", near(pr["2"]["pct_mean"],6.5,0.05) and near(pr["20"]["pct_mean"],13.6,0.05))
ge=EX["gate_effect"]["per_ratio"]
ck("gate_cost_range", near(min(ge[rt]["cost_increase_pct_mean"] for rt in ge),1.8,0.05) and near(max(ge[rt]["cost_increase_pct_mean"] for rt in ge),2.9,0.05))
ck("gate_fa", near(ge["5"]["false30_ungated"],0.097,0.0005) and near(ge["5"]["false30_gated"],0.046,0.0005))
ck("gate_pass", near(EX["gate_pass_rate_mean"],0.2267,0.0005))
gp=EX["gate_pass_rates_pe"]
ck("gate_pass_range", near(min(v["pass_rate"] for v in gp.values()),0.208,0.0005) and near(max(v["pass_rate"] for v in gp.values()),0.257,0.0005))
ck("gate_cand_range", min(v["n_cand"] for v in gp.values())==1263 and max(v["n_cand"] for v in gp.values())==1860)
ck("wear_avoid", near(P["per_ratio"]["5"]["reactive"]["mean"]["wear_fails"],76.7,0.05) and near(P["per_ratio"]["5"]["rul"]["mean"]["wear_fails"],1.3,0.05))
ck("avail_rul", near(P["per_ratio"]["5"]["rul"]["mean"]["avail"],0.813,0.0005) and near(P["per_ratio"]["5"]["reactive"]["mean"]["avail"],0.794,0.0005))
ck("f1000", near(P["per_ratio"]["5"]["rul"]["mean"]["fails_per_1000"],21.7,0.05) and near(P["per_ratio"]["5"]["reactive"]["mean"]["fails_per_1000"],25.9,0.05))
# ablation table
for rt in ["2","5","10","20"]:
    f1=ge[rt]["false30_ungated"]; f2=ge[rt]["false30_gated"]
    ptxt="$6.1{\\times}10^{-5}$" if pr[rt]["wilcoxon_p"] < 1e-4 else "%.4f"%pr[rt]["wilcoxon_p"]
    row="%s & %.1f $\\pm$ %.1f & %s & %.1f $\\pm$ %.1f & %.3f & %.3f"%(rt,pr[rt]["pct_mean"],pr[rt]["pct_sd"],ptxt,ge[rt]["cost_increase_pct_mean"],ge[rt]["cost_increase_pct_sd"],f1,f2)
    ck("abl_r"+rt, row in tex)
# shap table
A=S["shap"]
gtn={'TWF':'tool wear','HDF':'air T, process T, speed','PWF':'speed, torque','OSF':'tool wear, torque, type','RNF':'none (random)'}
for t in ["TWF","HDF","PWF","OSF","RNF"]:
    d=A[t]
    hm="%.2f $\\pm$ %.2f"%(d["hit_mean"],np.std(d["hit_rates"])) if d["hit_mean"] is not None else "--"
    row="%s & %s & %d & %s & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f"%(t,gtn[t],d["n_pos"],hm,d["jaccard_top3"],d["jaccard_sd"],d["spearman"],d["spearman_sd"])
    ck("shaprow_"+t, row in tex)
ck("shap_hits", A["TWF"]["hit_mean"]==1.0 and A["HDF"]["hit_mean"]==1.0 and A["PWF"]["hit_mean"]==1.0 and near(A["OSF"]["hit_mean"],2/3,1e-9))
# Q1 revision claim-boundary and robustness checks
ck("q1_title", "A Benchmark-Anchored Decision Environment with Mechanism-Aligned Gating for Predictive Maintenance" in tex)
ck("q1_no_benchmark_calibrated", "benchmark-calibrated" not in tex)
ck(
    "q1_partial_calibration",
    re.search(r"published\s+synthetic\s+benchmark", tex) is not None
    and re.search(r"partially\s+calibrated\s+simulator", tex) is not None
    and re.search(r"plant\s+temporal\s+calibration", tex) is not None,
)
ck("q1_cmapss_engine_ci", "-0.29" in tex and "-1.21" in tex and "0.60" in tex and "-0.96" in tex and "-1.63" in tex and "$-0.30$" in tex)
rob=Q1R["policy_robustness"]["by_ratio"]
rob_names=[("rul_base","RUL-aware"),("shap_gate","TreeSHAP gate"),("strict_rul_selected","Strict-RUL"),("raw_wear_selected","Raw-wear"),("wear_torque_selected","Wear-torque"),("periodic_fine_selected","Fine periodic"),("oracle_wear_final","Oracle wear")]
for rt in ["2","5","10","20"]:
    cells=[]
    for key,_ in rob_names:
        d=rob[rt][key]
        cells.append("%.3f (%.3f)"%(d["cost_rate"]["mean"],d["false_rate"]["mean"]))
    row=rt+" & "+" & ".join(cells)+" \\\\"
    ck("q1rob_row_"+rt, row in tex)
cg=Q1R["candidate_gate_validation"]
ck("q1_candidate_gate", near(cg["shap_gate"]["precision"]["mean"],0.802,0.0006) and near(cg["raw_wear_matched"]["precision"]["mean"],0.954,0.0006) and near(cg["wear_torque_matched"]["precision"]["mean"],0.879,0.0006) and "0.879\\pm0.034" in tex)
ths=Q1["false_threshold_sensitivity"]["5"]
ck("q1_false_thresholds", near(ths["10"]["rul_base_false_rate"]["mean"],0.943,0.0006) and near(ths["10"]["shap_gate_false_rate"]["mean"],0.783,0.0006) and "0.943 versus 0.783" in tex and "0.000 versus 0.000 under 40 cycles" in tex)
hwv=HW["validation_pool"]
hwr=HW["by_ratio"]
ck("q1_high_wear_pool", near(hwv["mean_episode_length"],24.4,0.05) and near(hwv["wear_mean"],135.2,0.05) and near(hwv["mode_rates_per_1000_cycles"]["TWF"],8.2,0.05) and near(hwv["mode_rates_per_1000_cycles"]["OSF"],13.7,0.05) and "high-wear warm-start stress test" in tex)
ck("q1_high_wear_costs", near(hwr["5"]["rul_base"]["cost_rate"]["mean"],0.168,0.0006) and near(hwr["5"]["shap_gate"]["cost_rate"]["mean"],0.184,0.0006) and near(hwr["20"]["rul_base"]["cost_rate"]["mean"],0.519,0.0006) and near(hwr["20"]["shap_gate"]["cost_rate"]["mean"],0.621,0.0006) and "0.168, 0.285, and 0.519" in tex and "0.184, 0.329, and 0.621" in tex)
ck("q1_high_wear_fa", near(hwr["5"]["rul_base"]["false_rate"]["mean"],0.070,0.0006) and near(hwr["5"]["shap_gate"]["false_rate"]["mean"],0.043,0.0006) and near(hwr["10"]["strict_rul_selected"]["false_rate"]["mean"],0.004,0.0006) and near(hwr["20"]["raw_wear_selected"]["false_rate"]["mean"],0.000,0.0006) and "0.070 to 0.043" in tex and "0.004 or 0.000" in tex)
msa=MSA["summary"]["by_ratio"]
ck("q1_model_seed_costs", near(msa["5"]["policy"]["rul"]["cost_rate"]["all_cells"]["mean"],0.1264,0.0001) and near(msa["5"]["policy"]["rul"]["cost_rate"]["diagonal"]["mean"],0.1265,0.0001) and near(msa["20"]["policy"]["rul"]["cost_rate"]["all_cells"]["mean"],0.4513,0.0001) and near(msa["20"]["policy"]["rul"]["cost_rate"]["diagonal"]["mean"],0.4518,0.0001) and "0.1264 versus 0.1265" in tex and "0.4513 versus 0.4518" in tex)
ck("q1_model_seed_variation", near(msa["5"]["policy"]["rul"]["cost_rate"]["pool_mean_sd"],0.0066,0.0001) and near(msa["5"]["policy"]["rul"]["cost_rate"]["model_mean_sd"],0.0001,0.0001) and near(msa["20"]["policy"]["rul"]["cost_rate"]["pool_mean_sd"],0.0259,0.0001) and near(msa["20"]["policy"]["rul"]["cost_rate"]["model_mean_sd"],0.0002,0.0001) and "0.0066, 0.0130, and 0.0259" in tex and "0.0001, 0.0001, and 0.0002" in tex)
ck("q1_model_seed_gate", near(msa["5"]["contrasts"]["rul_gated_minus_rul_cost"]["mean"],0.0023,0.0001) and near(msa["20"]["contrasts"]["rul_gated_minus_rul_cost"]["mean"],0.0136,0.0001) and near(msa["5"]["policy"]["rul"]["false30_rate"]["all_cells"]["mean"],0.099,0.0006) and near(msa["5"]["policy"]["rul_gated"]["false30_rate"]["all_cells"]["mean"],0.045,0.0006) and near(msa["5"]["contrasts"]["rul_gated_minus_rul_n_prev"]["mean"],-21.4,0.05) and "0.0023, 0.0060, and 0.0136" in tex and "0.099 to 0.045" in tex and "21.4 per 400-episode pool" in tex)
print()
print("FAILURES:",fails if fails else "none")
sys.exit(1 if fails else 0)

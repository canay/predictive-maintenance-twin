import os, sys, json, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time(); MAX_SEC = 30
FIG = BASE + "/figures"
os.makedirs(FIG, exist_ok=True)
# Okabe-Ito colorblind-safe palette
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#F0E442"]
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", name)

def done(name):
    return os.path.exists(f"{FIG}/{name}.png")

def budget():
    if time.time() - T0 > MAX_SEC:
        print("BUDGET"); sys.exit(2)

MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FE = ["Type", "Air T", "Proc. T", "Speed", "Torque", "Wear"]

# (a) architecture diagram -------------------------------------------------
if not done("fig_architecture"):
    budget()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.axis("off")
    def box(x, y, w, h, text, fc="#eef3f8", ec="#0072B2", fs=8.2, lw=1.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)
    def arrow(x0, y0, x1, y1, text=None, color="#444444", style="-|>", ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.3, linestyle=ls))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.025, text, ha="center",
                    va="bottom", fontsize=7.2, color=color)
    # data layer
    box(0.01, 0.62, 0.17, 0.30, "AI4I 2020\nreal sensor records\n(10,000 ops,\n5 failure modes)", fc="#fdf3e3", ec="#E69F00")
    box(0.01, 0.10, 0.17, 0.30, "Calibrated\nrun-to-failure\nsimulator\n(fitted to AI4I)", fc="#fdf3e3", ec="#E69F00")
    arrow(0.095, 0.62, 0.095, 0.42, "calibration", color="#E69F00")
    # agents
    box(0.24, 0.40, 0.155, 0.34, "A1 Detector\ncondition monitoring\nfailure risk $\\hat{p}(x_t)$")
    box(0.435, 0.40, 0.155, 0.34, "A2 Prognostic\ndegradation model\nRUL $\\widehat{R}(x_t)$")
    box(0.63, 0.40, 0.155, 0.34, "A3 Planner\ncost-aware policy\nact if $\\widehat{R}\\leq h$")
    box(0.825, 0.40, 0.16, 0.34, "A4 Explainer\nTreeSHAP attribution\n+ stability gate")
    arrow(0.18, 0.77, 0.30, 0.74)
    arrow(0.18, 0.25, 0.51, 0.40)
    arrow(0.395, 0.57, 0.435, 0.57, "risk")
    arrow(0.59, 0.57, 0.63, 0.57, "RUL")
    arrow(0.785, 0.57, 0.825, 0.57, "trigger")
    # gate feedback
    ax.annotate("", xy=(0.71, 0.40), xytext=(0.90, 0.40),
                arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=1.5,
                                connectionstyle="arc3,rad=0.35"))
    ax.text(0.805, 0.20, "explanation gate:\nact only if wear-driven\nattribution is consistent",
            ha="center", fontsize=7.4, color="#D55E00")
    box(0.40, 0.02, 0.23, 0.14, "maintenance action\n(replace / continue)", fc="#e8f5ee", ec="#009E73")
    ax.annotate("", xy=(0.63, 0.09), xytext=(0.71, 0.36),
                arrowprops=dict(arrowstyle="-|>", color="#009E73", lw=1.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig_architecture")

# (b) risk performance + calibration --------------------------------------
if not done("fig_risk"):
    budget()
    targets = ["Machine failure"] + MODES
    models = ["LR", "RF", "HGB"]
    res = {}
    for m in models:
        for t in targets:
            fs = glob.glob(BASE + f"/results/risk/u_{m}_{t.replace(' ', '_')}_*.json")
            rs = [json.load(open(f)) for f in fs]
            res[(m, t)] = rs
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    x = np.arange(len(targets)); w = 0.26
    for ax, met, lab in [(axes[0], "auroc", "AUROC"), (axes[1], "ap", "Average precision")]:
        for i, m in enumerate(models):
            mu = [np.mean([r[met] for r in res[(m, t)]]) for t in targets]
            sd = [np.std([r[met] for r in res[(m, t)]]) for t in targets]
            ax.bar(x + (i - 1) * w, mu, w, yerr=sd, color=OI[i], label=m, capsize=1.5,
                   error_kw=dict(lw=0.7))
        ax.set_xticks(x); ax.set_xticklabels(["All"] + MODES, rotation=45)
        ax.set_ylabel(lab); ax.set_ylim(0, 1.02)
    axes[0].legend(frameon=False, loc="lower left", ncol=1)
    # reliability diagram, RF / overall failure, seed 0
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    df = load_ai4i(); X = df[FEATS].values; y = df["Machine failure"].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced", n_jobs=2, random_state=0).fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    bins = np.linspace(0, 1, 11); mids, frac, conf = [], [], []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < 9 else (p >= bins[i])
        if m.sum() > 5:
            mids.append((bins[i] + bins[i + 1]) / 2)
            frac.append(yte[m].mean()); conf.append(p[m].mean())
    axes[2].plot([0, 1], [0, 1], "--", color="#888888", lw=0.9)
    axes[2].plot(conf, frac, "o-", color=OI[1], ms=3.5, lw=1.2)
    axes[2].set_xlabel("Predicted probability"); axes[2].set_ylabel("Observed frequency")
    fig.tight_layout()
    save(fig, "fig_risk")

# (c) simulator calibration fit -------------------------------------------
if not done("fig_calibration"):
    budget()
    from simulator import fit_params, fit_pools, gen_pool, pool_to_arrays
    df = load_ai4i()
    P = fit_params(df); POOLS = fit_pools(df)
    eps = gen_pool(12345, 600, P, POOLS)
    Xs, _, _ = pool_to_arrays(eps)
    cal = json.load(open(BASE + "/results/calibration.json"))
    real = [df["Air temperature [K]"].values, df["Process temperature [K]"].values,
            df["Rotational speed [rpm]"].values, df["Torque [Nm]"].values,
            df["Tool wear [min]"].values]
    sim = [Xs[:, 1], Xs[:, 2], Xs[:, 3], Xs[:, 4], Xs[:, 5]]
    names = ["Air temperature [K]", "Process temp. [K]", "Rot. speed [rpm]",
             "Torque [Nm]", "Tool wear [min]"]
    keys = ["air", "proc", "rpm", "tq", "wear"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.8))
    for i in range(5):
        ax = axes.flat[i]
        lo = min(real[i].min(), sim[i].min()); hi = max(real[i].max(), sim[i].max())
        b = np.linspace(lo, hi, 40)
        ax.hist(real[i], bins=b, density=True, alpha=0.55, color=OI[0], label="real")
        ax.hist(sim[i], bins=b, density=True, alpha=0.55, color=OI[1], label="simulated")
        ax.set_xlabel(names[i]); ax.set_ylabel("Density")
        ax.text(0.97, 0.95, "KS=%.3f" % cal[keys[i]]["ks"], transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.5)
    ax = axes.flat[5]
    xm = np.arange(5); w = 0.38
    rs = [cal["failure_rates_real_per_1000_rows"][m] for m in MODES]
    ss = [cal["failure_rates_per_1000_cycles"][m] for m in MODES]
    ax.bar(xm - w / 2, rs, w, color=OI[0], label="real")
    ax.bar(xm + w / 2, ss, w, color=OI[1], label="simulated")
    ax.set_xticks(xm); ax.set_xticklabels(MODES, rotation=45)
    ax.set_ylabel("Failures per 1000 cycles")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_calibration")

# (d) RUL quality / alpha-lambda -------------------------------------------
if not done("fig_rul"):
    budget()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    lam = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i, m in enumerate(["HGBR", "MLP"]):
        rs = [json.load(open(f)) for f in sorted(glob.glob(BASE + f"/results/rul/u2_{m}_*.json"))]
        A = np.array([[r["alpha_lambda"][str(l)] for l in lam] for r in rs])
        axes[0].errorbar(lam, A.mean(0), yerr=A.std(0), color=OI[i], marker="o", ms=3,
                         lw=1.2, capsize=2, label=m.replace("HGBR", "HGB"))
    axes[0].set_xlabel("Relative life position $\\lambda$")
    axes[0].set_ylabel("$\\alpha$-$\\lambda$ accuracy ($\\alpha$=0.3)")
    axes[0].set_ylim(0, 1.05); axes[0].legend(frameon=False)
    z = np.load(BASE + "/results/rul/preds_HGBR_s0.npz", allow_pickle=True)
    pred, rul, eid = z["pred"], z["rul"], z["eid"]
    shown = 0
    for e in np.unique(eid):
        m = eid == e; T = int(m.sum())
        if T < 40 or shown >= 3:
            continue
        t = np.arange(T)
        axes[1].plot(t, rul[m], color="#888888", lw=1.0,
                     label="true RUL" if shown == 0 else None)
        axes[1].plot(t, pred[m], color=OI[shown], lw=1.2,
                     label="predicted (ep. %d)" % (shown + 1))
        shown += 1
    axes[1].set_xlabel("Cycle"); axes[1].set_ylabel("RUL [cycles]")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_rul")

# (e) policy cost sweep (headline) ------------------------------------------
if not done("fig_policy"):
    budget()
    pe = json.load(open(BASE + "/results/policy_eval.json"))
    ratios = [2, 5, 10, 20]
    pols = [("reactive", "Reactive", OI[6], "--"), ("periodic", "Periodic (best $K$)", OI[2], "-"),
            ("risk", "Predictive: risk only", OI[0], "-"), ("rul", "Predictive: RUL-aware", OI[3], "-"),
            ("rul_gated", "RUL-aware + expl. gate", OI[4], "-")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for key, lab, c, ls in pols:
        mu = [pe["per_ratio"][str(r)][key]["mean"]["cost_rate"] for r in ratios]
        sd = [pe["per_ratio"][str(r)][key]["sd"]["cost_rate"] for r in ratios]
        axes[0].errorbar(ratios, mu, yerr=sd, label=lab, color=c, ls=ls, marker="o",
                         ms=3, lw=1.3, capsize=2)
        av = [pe["per_ratio"][str(r)][key]["mean"]["avail"] for r in ratios]
        avs = [pe["per_ratio"][str(r)][key]["sd"]["avail"] for r in ratios]
        axes[1].errorbar(ratios, av, yerr=avs, color=c, ls=ls, marker="o", ms=3,
                         lw=1.3, capsize=2)
    axes[0].set_xlabel("Cost ratio $c_c/c_p$"); axes[0].set_ylabel("Cost per operating cycle")
    axes[0].set_xscale("log"); axes[0].set_xticks(ratios); axes[0].set_xticklabels(ratios)
    axes[0].legend(frameon=False, fontsize=6.8, loc="upper left")
    axes[1].set_xlabel("Cost ratio $c_c/c_p$"); axes[1].set_ylabel("Availability")
    axes[1].set_xscale("log"); axes[1].set_xticks(ratios); axes[1].set_xticklabels(ratios)
    fig.tight_layout()
    save(fig, "fig_policy")

# (f) ablation deltas --------------------------------------------------------
if not done("fig_ablation"):
    budget()
    pe = json.load(open(BASE + "/results/policy_eval.json"))
    ratios = [2, 5, 10, 20]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    # panel 1: cost delta risk-only minus RUL-aware (value of prognostic layer)
    x = np.arange(len(ratios)); w = 0.6
    d1 = []
    for r in ratios:
        a = np.array(pe["per_ratio"][str(r)]["risk"]["per_seed"]["cost_rate"])
        b = np.array(pe["per_ratio"][str(r)]["rul"]["per_seed"]["cost_rate"])
        d1.append(100 * (a - b) / a)
    axes[0].bar(x, [d.mean() for d in d1], w, yerr=[d.std() for d in d1], color=OI[3],
                capsize=2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ratios)
    axes[0].set_xlabel("Cost ratio $c_c/c_p$")
    axes[0].set_ylabel("Cost reduction from\nprognostic layer [%]")
    # panel 2: gate effect on cost
    d2 = []
    for r in ratios:
        a = np.array(pe["per_ratio"][str(r)]["rul_gated"]["per_seed"]["cost_rate"])
        b = np.array(pe["per_ratio"][str(r)]["rul"]["per_seed"]["cost_rate"])
        d2.append(100 * (a - b) / b)
    axes[1].bar(x, [d.mean() for d in d2], w, yerr=[d.std() for d in d2], color=OI[4],
                capsize=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(ratios)
    axes[1].set_xlabel("Cost ratio $c_c/c_p$")
    axes[1].set_ylabel("Cost increase from\nexplanation gate [%]")
    # panel 3: false action rate, gated vs ungated
    f_u, f_g, fu_sd, fg_sd = [], [], [], []
    for r in ratios:
        a = np.array(pe["per_ratio"][str(r)]["rul"]["per_seed"]["false30_rate"])
        b = np.array(pe["per_ratio"][str(r)]["rul_gated"]["per_seed"]["false30_rate"])
        f_u.append(a.mean()); fu_sd.append(a.std())
        f_g.append(b.mean()); fg_sd.append(b.std())
    w2 = 0.38
    axes[2].bar(x - w2 / 2, f_u, w2, yerr=fu_sd, color=OI[3], capsize=2, label="RUL-aware")
    axes[2].bar(x + w2 / 2, f_g, w2, yerr=fg_sd, color=OI[4], capsize=2, label="+ expl. gate")
    axes[2].set_xticks(x); axes[2].set_xticklabels(ratios)
    axes[2].set_xlabel("Cost ratio $c_c/c_p$")
    axes[2].set_ylabel("False maintenance\naction rate")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_ablation")

# (g) attribution hit rate ----------------------------------------------------
if not done("fig_attribution"):
    budget()
    sm = json.load(open(BASE + "/results/shap_summary.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    mm = ["TWF", "HDF", "PWF", "OSF"]
    x = np.arange(len(mm)); w = 0.6
    hits = [sm[m]["hit_mean"] for m in mm]
    hsd = [np.std(sm[m]["hit_rates"]) for m in mm]
    axes[0].bar(x, hits, w, yerr=hsd, color=OI[0], capsize=2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(mm)
    axes[0].set_ylabel("Attribution hit rate")
    axes[0].set_ylim(0, 1.08)
    axes[0].axhline(1.0, ls="--", lw=0.8, color="#888888")
    # heatmap of mean |SHAP| per mode (positives statistic)
    M = np.array([sm[m]["mean_imp_pos"] for m in mm + ["RNF"]])
    Mn = M / M.sum(1, keepdims=True)
    im = axes[1].imshow(Mn, cmap="cividis", aspect="auto")
    axes[1].set_xticks(range(6)); axes[1].set_xticklabels(FE, rotation=45, ha="right")
    axes[1].set_yticks(range(5)); axes[1].set_yticklabels(mm + ["RNF"])
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046)
    cb.set_label("Normalized mean $|$SHAP$|$", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_attribution")

print("ALL FIGS DONE")


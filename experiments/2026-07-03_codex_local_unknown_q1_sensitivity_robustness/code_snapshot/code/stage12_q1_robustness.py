import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import shap
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, FEATS, jdump, load_ai4i


DD = BASE + "/data"
OUT = BASE + "/results"
CMAPSS = OUT + "/cmapss_external_vps_20260614_121237"
SP = json.load(open(OUT + "/sim_params.json"))
POLICY = json.load(open(OUT + "/policy_eval.json"))
EVAL_SEEDS = list(POLICY.get("eval_seeds", range(5)))
MEAN_INC = float(np.dot(SP["inc_vals"], SP["inc_probs"]))
CP, CD_RATE, DP, DC = 1.0, 0.05, 2, 10
RATIOS = [2, 5, 10, 20]
FALSE_THRESHOLDS = [10, 20, 30, 40]


def load_pool(tag, seed):
    z = np.load(f"{DD}/pool2_{tag}_{seed}.npz", allow_pickle=True)
    g = np.load(f"{OUT}/gate_{tag}_{seed}.npz", allow_pickle=True)
    gate_full = np.zeros(len(z["X"]), bool)
    gate_full[g["cand"]] = g["gate"]
    eid = z["eid"].astype(int)
    bounds = np.searchsorted(eid, np.arange(eid.max() + 2))
    eps = []
    for e in range(int(eid.max()) + 1):
        a, b = bounds[e], bounds[e + 1]
        eps.append(
            dict(
                T=b - a,
                risk=g["risk"][a:b],
                pred_rul=g["rul"][a:b],
                true_rul=z["rul"][a:b],
                gate=gate_full[a:b],
                wear=z["X"][a:b, 5],
                torque=z["X"][a:b, 4],
                wtwf=float(z["wtwf"][e]),
                mode=str(z["modes"][e]),
            )
        )
    return eps


def terminal_outcome(ep):
    return dict(prev=0, fail=1, op=ep["T"], down=DC, mode=ep["mode"], rem=None)


def preventive_outcome(ep, t):
    rem = float((ep["wtwf"] - ep["wear"][t]) / MEAN_INC)
    return dict(prev=1, fail=0, op=t + 1, down=DP, mode=None, rem=rem)


def first_trigger(ep, policy, param=None):
    T = ep["T"]
    if policy == "reactive":
        return None
    if policy in ("periodic", "periodic_fine"):
        return int(param - 1) if param - 1 <= T - 2 else None
    if policy == "rul":
        trig = ep["pred_rul"] <= param
    elif policy == "rul_gated":
        trig = (ep["pred_rul"] <= param) & ep["gate"]
    elif policy == "strict_rul":
        trig = ep["pred_rul"] <= param
    elif policy == "raw_wear":
        trig = ep["wear"] >= param
    elif policy == "wear_torque":
        trig = (ep["wear"] * ep["torque"]) >= param
    elif policy == "oracle_wear_final":
        if ep["mode"] in ("TWF", "OSF") and T >= 2:
            return T - 2
        return None
    else:
        raise ValueError(policy)
    idx = np.where(trig[: T - 1])[0]
    return int(idx[0]) if len(idx) else None


def eval_policy(eps, policy, param, ratio, false_threshold=30):
    outs = []
    for ep in eps:
        t = first_trigger(ep, policy, param)
        outs.append(terminal_outcome(ep) if t is None else preventive_outcome(ep, t))
    op = sum(o["op"] for o in outs)
    down = sum(o["down"] for o in outs)
    n_prev = sum(o["prev"] for o in outs)
    n_fail = sum(o["fail"] for o in outs)
    cost = n_prev * CP + n_fail * ratio * CP + CD_RATE * down
    false_n = sum(1 for o in outs if o["prev"] and o["rem"] is not None and o["rem"] > false_threshold)
    rems = [o["rem"] for o in outs if o["prev"] and o["rem"] is not None]
    return dict(
        cost_rate=float(cost / op),
        avail=float(op / (op + down)),
        n_prev=float(n_prev),
        n_fail=float(n_fail),
        fails_per_1000=float(1000 * n_fail / op),
        false_rate=float(false_n / n_prev) if n_prev else 0.0,
        mean_rem=float(np.mean(rems)) if rems else 0.0,
        op=float(op),
        total_cost=float(cost),
    )


def mean_sd(vals):
    vals = np.asarray(vals, float)
    return dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))


def bootstrap_ci(vals, n_boot=10000, seed=123):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return dict(lo=float(np.quantile(draws, 0.025)), hi=float(np.quantile(draws, 0.975)))


def aggregate(rows):
    out = {}
    for m in ["cost_rate", "avail", "n_prev", "n_fail", "fails_per_1000", "false_rate", "mean_rem"]:
        vals = [r[m] for r in rows]
        out[m] = mean_sd(vals)
        out[m]["ci95"] = bootstrap_ci(vals, seed=1000 + len(m))
    return out


def holm_adjust(pvals):
    valid = [(i, p) for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    out = [None] * len(pvals)
    m = len(valid)
    prev = 0.0
    for rank, (i, p) in enumerate(sorted(valid, key=lambda x: x[1])):
        adj = min(1.0, (m - rank) * p)
        prev = max(prev, adj)
        out[i] = prev
    return out


def select_params(validation_eps, ratio):
    wt_values = np.concatenate([ep["wear"] * ep["torque"] for ep in validation_eps])
    wt_grid = sorted(set(float(round(q / 250.0) * 250.0) for q in np.quantile(wt_values, np.linspace(0.50, 0.98, 13))))
    grids = {
        "periodic_fine": list(range(10, 121, 5)),
        "strict_rul": [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32],
        "raw_wear": list(range(40, 241, 10)),
        "wear_torque": wt_grid,
    }
    selected = {}
    for policy, grid in grids.items():
        vals = [(eval_policy(validation_eps, policy, g, ratio)["cost_rate"], g) for g in grid]
        selected[policy] = min(vals, key=lambda x: x[0])[1]
    return selected


def policy_robustness():
    policy = POLICY
    validation = load_pool("pv", 0)
    pools = [load_pool("pe", s) for s in EVAL_SEEDS]
    block = {"selection": {}, "by_ratio": {}, "paired_tests": {}}
    csv_rows = []
    for ratio in RATIOS:
        rkey = str(ratio)
        selected = select_params(validation, ratio)
        selected["rul"] = int(policy["selection"][rkey]["rul"])
        block["selection"][rkey] = selected
        variants = [
            ("rul_base", "rul", selected["rul"]),
            ("shap_gate", "rul_gated", selected["rul"]),
            ("strict_rul_selected", "strict_rul", selected["strict_rul"]),
            ("raw_wear_selected", "raw_wear", selected["raw_wear"]),
            ("wear_torque_selected", "wear_torque", selected["wear_torque"]),
            ("periodic_fine_selected", "periodic_fine", selected["periodic_fine"]),
            ("oracle_wear_final", "oracle_wear_final", None),
        ]
        rows_by_variant = {}
        for name, policy_name, param in variants:
            rows = [eval_policy(eps, policy_name, param, ratio) for eps in pools]
            rows_by_variant[name] = rows
            for seed, row in zip(EVAL_SEEDS, rows):
                csv_rows.append({"ratio": ratio, "seed": seed, "analysis": "policy_robustness", "variant": name, **row})
        block["by_ratio"][rkey] = {name: aggregate(rows) for name, rows in rows_by_variant.items()}

        base = np.array([r["cost_rate"] for r in rows_by_variant["rul_base"]])
        tests, pvals = {}, []
        for name in [v[0] for v in variants if v[0] != "rul_base"]:
            vals = np.array([r["cost_rate"] for r in rows_by_variant[name]])
            delta = vals - base
            try:
                p = float(wilcoxon(vals, base).pvalue) if np.any(delta != 0) else 1.0
            except ValueError:
                p = None
            tests[name] = dict(
                mean_delta_cost_rate=float(np.mean(delta)),
                delta_ci95=bootstrap_ci(delta, seed=2000 + ratio),
                wilcoxon_p=p,
            )
            pvals.append(p)
        for name, adj in zip(tests.keys(), holm_adjust(pvals)):
            tests[name]["holm_p"] = adj
        block["paired_tests"][rkey] = tests
    return block, csv_rows


def candidate_metrics(mask, label):
    mask = np.asarray(mask, bool)
    label = np.asarray(label, bool)
    tp = int(np.sum(mask & label))
    pred = int(np.sum(mask))
    pos = int(np.sum(label))
    precision = tp / pred if pred else 0.0
    recall = tp / pos if pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        pass_rate=float(np.mean(mask)) if len(mask) else 0.0,
        tp=float(tp),
        pred=float(pred),
        positives=float(pos),
        n=float(len(mask)),
    )


def topk_mask(score, k):
    if k <= 0:
        return np.zeros(len(score), bool)
    idx = np.argsort(-np.asarray(score, float))[:k]
    mask = np.zeros(len(score), bool)
    mask[idx] = True
    return mask


def candidate_gate_validation():
    rows_by_filter = {k: [] for k in ["shap_gate", "strict_rul8", "raw_wear_matched", "wear_torque_matched", "random_matched"]}
    for seed in EVAL_SEEDS:
        z = np.load(f"{DD}/pool2_pe_{seed}.npz", allow_pickle=True)
        g = np.load(f"{OUT}/gate_pe_{seed}.npz", allow_pickle=True)
        cand = g["cand"]
        eid = z["eid"].astype(int)[cand]
        modes = z["modes"][eid].astype(str)
        wear = z["X"][cand, 5]
        torque = z["X"][cand, 4]
        rem_wear = (z["wtwf"][eid] - wear) / MEAN_INC
        label = np.isin(modes, ["TWF", "OSF"]) & (rem_wear <= 30)
        shap_mask = g["gate"].astype(bool)
        k = int(np.sum(shap_mask))
        masks = {
            "shap_gate": shap_mask,
            "strict_rul8": g["rul"][cand] <= 8,
            "raw_wear_matched": topk_mask(wear, k),
            "wear_torque_matched": topk_mask(wear * torque, k),
        }
        rng = np.random.default_rng(120000 + seed)
        random_stats = []
        for _ in range(200):
            random_stats.append(candidate_metrics(rng.random(len(cand)) < (k / len(cand)), label))
        for name, mask in masks.items():
            row = candidate_metrics(mask, label)
            row["seed"] = seed
            rows_by_filter[name].append(row)
        avg_random = {m: float(np.mean([r[m] for r in random_stats])) for m in random_stats[0]}
        avg_random["seed"] = seed
        rows_by_filter["random_matched"].append(avg_random)
    return {
        name: {m: mean_sd([r[m] for r in rows]) for m in ["precision", "recall", "f1", "pass_rate", "tp", "pred", "positives", "n"]}
        for name, rows in rows_by_filter.items()
    }


def shap_values_positive(explainer, x):
    vals = explainer.shap_values(x)
    if isinstance(vals, list):
        return vals[1]
    if getattr(vals, "ndim", 0) == 3:
        return vals[..., 1]
    return vals


def heldout_ai4i_gate_audit():
    df = load_ai4i()
    x = df[FEATS].to_numpy()
    y = df["Machine failure"].to_numpy(int)
    wear_failure = ((df["TWF"].to_numpy(int) == 1) | (df["OSF"].to_numpy(int) == 1))
    rows = []
    for seed in range(5):
        idx = np.arange(len(df))
        train_idx, tmp_idx, y_train, y_tmp = train_test_split(
            idx, y, test_size=0.4, stratify=y, random_state=500 + seed
        )
        val_idx, test_idx, y_val, y_test = train_test_split(
            tmp_idx, y_tmp, test_size=0.5, stratify=y_tmp, random_state=700 + seed
        )
        models = [
            RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=2,
                random_state=100 + g + 10 * seed,
            ).fit(x[train_idx], y_train)
            for g in range(3)
        ]

        def split_metrics(split_idx, split_name):
            failure_idx = split_idx[y[split_idx] == 1]
            votes = []
            for model in models:
                vals = np.abs(shap_values_positive(shap.TreeExplainer(model), x[failure_idx]))
                top2 = np.argsort(-vals, axis=1)[:, :2]
                votes.append(np.any(top2 == 5, axis=1))
            gate = np.vstack(votes).sum(axis=0) >= 2
            label = wear_failure[failure_idx]
            row = candidate_metrics(gate, label)
            row.update(
                seed=seed,
                split=split_name,
                n_train=int(len(train_idx)),
                n_val=int(len(val_idx)),
                n_test=int(len(test_idx)),
                failure_rows=int(len(failure_idx)),
            )
            return row

        rows.append(split_metrics(val_idx, "validation"))
        rows.append(split_metrics(test_idx, "test"))

    out = {}
    for split in ["validation", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        out[split] = {
            m: mean_sd([r[m] for r in split_rows])
            for m in ["precision", "recall", "f1", "pass_rate", "tp", "pred", "positives", "n", "failure_rows"]
        }
    return out


def cmapss_context():
    out = {}
    pred_dir = CMAPSS + "/predictions"
    for dataset in ["FD001", "FD002", "FD003", "FD004"]:
        out[dataset] = {}
        hgb_preds = []
        for seed in [0, 1, 2]:
            hgb_preds.append(pd.read_csv(f"{pred_dir}/{dataset}_hgb_seed{seed}.csv").sort_values("unit"))
        hgb_mean_pred = np.mean([p["pred_rul"].to_numpy(float) for p in hgb_preds], axis=0)
        true = hgb_preds[0]["true_rul"].to_numpy(float)
        unit = hgb_preds[0]["unit"].to_numpy(int)
        hgb_err = hgb_mean_pred - true
        out[dataset]["hgb_seed_mean"] = dict(
            n_engines=int(len(unit)),
            rmse=float(np.sqrt(np.mean(hgb_err**2))),
            mae=float(np.mean(np.abs(hgb_err))),
        )
        for model in ["rf", "ridge"]:
            if model == "rf":
                preds = [pd.read_csv(f"{pred_dir}/{dataset}_rf_seed{seed}.csv").sort_values("unit") for seed in [0, 1, 2]]
                comp_pred = np.mean([p["pred_rul"].to_numpy(float) for p in preds], axis=0)
            else:
                comp_pred = pd.read_csv(f"{pred_dir}/{dataset}_ridge_seed0.csv").sort_values("unit")["pred_rul"].to_numpy(float)
            comp_err = comp_pred - true
            delta_abs = np.abs(hgb_err) - np.abs(comp_err)
            delta_sq = hgb_err**2 - comp_err**2
            rng = np.random.default_rng(3000 + len(dataset) + len(model))
            idx = rng.integers(0, len(true), size=(10000, len(true)))
            hgb_rmse = np.sqrt(np.mean(hgb_err[idx] ** 2, axis=1))
            comp_rmse = np.sqrt(np.mean(comp_err[idx] ** 2, axis=1))
            out[dataset][f"hgb_vs_{model}"] = dict(
                comparator_rmse=float(np.sqrt(np.mean(comp_err**2))),
                comparator_mae=float(np.mean(np.abs(comp_err))),
                rmse_delta_hgb_minus_comparator=float(out[dataset]["hgb_seed_mean"]["rmse"] - np.sqrt(np.mean(comp_err**2))),
                rmse_delta_ci95=dict(
                    lo=float(np.quantile(hgb_rmse - comp_rmse, 0.025)),
                    hi=float(np.quantile(hgb_rmse - comp_rmse, 0.975)),
                ),
                mean_abs_error_delta_hgb_minus_comparator=float(np.mean(delta_abs)),
                mean_squared_error_delta_hgb_minus_comparator=float(np.mean(delta_sq)),
            )
    return out


def main():
    policy_block, csv_rows = policy_robustness()
    robust = {
        "description": "Additional Q1 robustness checks derived from saved pools and external validation predictions.",
        "policy_robustness": policy_block,
        "candidate_gate_validation": candidate_gate_validation(),
        "heldout_ai4i_gate_audit": heldout_ai4i_gate_audit(),
        "cmapss_engine_level_context": cmapss_context(),
        "note": "All policy robustness analyses remain simulator-bound and are not plant-deployment evidence.",
    }
    jdump(robust, OUT + "/q1_robustness.json")
    with open(OUT + "/q1_robustness.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ratio",
            "seed",
            "analysis",
            "variant",
            "cost_rate",
            "avail",
            "n_prev",
            "n_fail",
            "fails_per_1000",
            "false_rate",
            "mean_rem",
            "op",
            "total_cost",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)
    print("wrote", OUT + "/q1_robustness.json")
    print(json.dumps(robust["candidate_gate_validation"], indent=1))


if __name__ == "__main__":
    main()

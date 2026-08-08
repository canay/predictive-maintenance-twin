import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, jdump


DD = BASE + "/data"
OUT = BASE + "/results"
SP = json.load(open(OUT + "/sim_params.json"))
POLICY = json.load(open(OUT + "/policy_eval.json"))
MEAN_INC = float(np.dot(SP["inc_vals"], SP["inc_probs"]))
CP, CD_RATE, DP, DC = 1.0, 0.05, 2, 10
RATIOS = [2, 5, 10, 20]
POLICIES = ["reactive", "periodic", "risk", "rul", "risk_gated", "rul_gated"]
POOL_SEEDS = list(POLICY.get("eval_seeds", range(5)))
MODEL_SEEDS = list(range(5))


def shap_values_positive(explainer, x):
    vals = explainer.shap_values(x)
    if isinstance(vals, list):
        return vals[1]
    if getattr(vals, "ndim", 0) == 3:
        return vals[..., 1]
    return vals


def load_pool(pool_seed):
    z = np.load(f"{DD}/pool2_pe_{pool_seed}.npz", allow_pickle=True)
    return {
        "X": z["X"],
        "eid": z["eid"].astype(int),
        "wear": z["X"][:, 5],
        "wtwf": z["wtwf"].astype(float),
        "modes": z["modes"].astype(str),
    }


def split_episodes(pool, risk, pred_rul, gate):
    eid = pool["eid"]
    bounds = np.searchsorted(eid, np.arange(eid.max() + 2))
    eps = []
    for e in range(int(eid.max()) + 1):
        a, b = bounds[e], bounds[e + 1]
        eps.append(
            {
                "T": b - a,
                "risk": risk[a:b],
                "rul": pred_rul[a:b],
                "gate": gate[a:b],
                "wear": pool["wear"][a:b],
                "wtwf": float(pool["wtwf"][e]),
                "mode": str(pool["modes"][e]),
            }
        )
    return eps


def wear_vote_for_cycles(explainers, x, idx):
    vote_full = np.zeros(len(x), dtype=bool)
    if len(idx) == 0:
        return vote_full
    votes = []
    xc = x[idx]
    for explainer in explainers:
        vals = np.abs(shap_values_positive(explainer, xc))
        top2 = np.argsort(-vals, axis=1)[:, :2]
        votes.append(np.any(top2 == 5, axis=1))
    vote_full[idx] = np.vstack(votes).sum(axis=0) >= 2
    return vote_full


def make_predictions(pool, model_seed, risk_models, rul_models):
    x = pool["X"]
    risk = risk_models[model_seed].predict_proba(x)[:, 1]
    pred_rul = rul_models[model_seed].predict(x)
    cand = np.where((pred_rul <= 16) | (risk >= 0.5))[0]
    return risk, pred_rul, cand


def ep_outcome(ep, policy, param):
    t = None
    T = ep["T"]
    if policy == "periodic":
        t = int(param - 1) if param - 1 <= T - 2 else None
    elif policy == "risk":
        idx = np.where((ep["risk"] >= param)[: T - 1])[0]
        t = int(idx[0]) if len(idx) else None
    elif policy == "rul":
        idx = np.where((ep["rul"] <= param)[: T - 1])[0]
        t = int(idx[0]) if len(idx) else None
    elif policy == "risk_gated":
        idx = np.where(((ep["risk"] >= param) & ep["gate"])[: T - 1])[0]
        t = int(idx[0]) if len(idx) else None
    elif policy == "rul_gated":
        idx = np.where(((ep["rul"] <= param) & ep["gate"])[: T - 1])[0]
        t = int(idx[0]) if len(idx) else None
    elif policy != "reactive":
        raise ValueError(policy)

    if t is None:
        return {"prev": 0, "fail": 1, "op": T, "down": DC, "mode": ep["mode"], "false30": 0, "rem": None}
    rem = float((ep["wtwf"] - ep["wear"][t]) / MEAN_INC)
    return {"prev": 1, "fail": 0, "op": t + 1, "down": DP, "mode": None, "false30": int(rem > 30), "rem": rem}


def evaluate(eps, policy, param, ratio):
    outcomes = [ep_outcome(ep, policy, param) for ep in eps]
    op = sum(o["op"] for o in outcomes)
    down = sum(o["down"] for o in outcomes)
    n_prev = sum(o["prev"] for o in outcomes)
    n_fail = sum(o["fail"] for o in outcomes)
    cost = n_prev * CP + n_fail * ratio * CP + CD_RATE * down
    wear_fails = sum(1 for o in outcomes if o["mode"] in ("TWF", "OSF"))
    false30 = sum(o["false30"] for o in outcomes)
    rems = [o["rem"] for o in outcomes if o["rem"] is not None]
    return {
        "cost_rate": float(cost / op),
        "avail": float(op / (op + down)),
        "n_prev": float(n_prev),
        "n_fail": float(n_fail),
        "fails_per_1000": float(1000 * n_fail / op),
        "wear_fails": float(wear_fails),
        "false30_rate": float(false30 / n_prev) if n_prev else 0.0,
        "mean_rem_at_prev": float(np.mean(rems)) if rems else 0.0,
        "op": float(op),
    }


def aggregate(vals):
    vals = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def summarize_cells(rows, metric):
    vals = np.array([r[metric] for r in rows], dtype=float)
    diag = np.array([r[metric] for r in rows if r["pool_seed"] % len(MODEL_SEEDS) == r["model_seed"]], dtype=float)
    fixed0 = np.array([r[metric] for r in rows if r["model_seed"] == 0], dtype=float)
    pool_index = {seed: i for i, seed in enumerate(POOL_SEEDS)}
    model_index = {seed: i for i, seed in enumerate(MODEL_SEEDS)}
    matrix = np.full((len(POOL_SEEDS), len(MODEL_SEEDS)), np.nan)
    for r in rows:
        matrix[pool_index[int(r["pool_seed"])], model_index[int(r["model_seed"])]] = r[metric]
    pool_means = np.nanmean(matrix, axis=1)
    model_means = np.nanmean(matrix, axis=0)
    residual = matrix - pool_means[:, None] - model_means[None, :] + np.nanmean(matrix)
    return {
        "all_cells": aggregate(vals),
        "diagonal": aggregate(diag),
        "fixed_model_0": aggregate(fixed0),
        "pool_mean_sd": float(np.nanstd(pool_means)),
        "model_mean_sd": float(np.nanstd(model_means)),
        "residual_sd": float(np.nanstd(residual)),
    }


def paired_delta(rows_by_policy, policy_a, policy_b, metric):
    a = rows_by_policy[policy_a]
    b = rows_by_policy[policy_b]
    key_to_b = {(r["pool_seed"], r["model_seed"]): r for r in b}
    vals = []
    for row in a:
        other = key_to_b[(row["pool_seed"], row["model_seed"])]
        vals.append(row[metric] - other[metric])
    return aggregate(vals)


def main():
    risk_models = {s: joblib.load(f"{DD}/riskrf_{s}.joblib") for s in MODEL_SEEDS}
    rul_models = {s: joblib.load(f"{DD}/rulmodel2_HGBR_{s}.joblib") for s in MODEL_SEEDS}
    explainers = [shap.TreeExplainer(joblib.load(f"{DD}/gaterf_{g}.joblib")) for g in range(3)]
    pools = {s: load_pool(s) for s in POOL_SEEDS}

    csv_rows = []
    gate_rows = []
    for pool_seed, pool in pools.items():
        pred_by_model = {}
        union_cand = []
        for model_seed in MODEL_SEEDS:
            risk, pred_rul, cand = make_predictions(pool, model_seed, risk_models, rul_models)
            pred_by_model[model_seed] = {"risk": risk, "rul": pred_rul, "cand": cand}
            union_cand.append(cand)
        union_cand = np.unique(np.concatenate(union_cand)) if union_cand else np.array([], dtype=int)
        wear_vote = wear_vote_for_cycles(explainers, pool["X"], union_cand)
        for model_seed in MODEL_SEEDS:
            pred = pred_by_model[model_seed]
            gate = np.zeros(len(pool["X"]), dtype=bool)
            gate[pred["cand"]] = wear_vote[pred["cand"]]
            eps = split_episodes(pool, pred["risk"], pred["rul"], gate)
            n_cand = len(pred["cand"])
            pass_rate = float(np.mean(gate[pred["cand"]])) if n_cand else 0.0
            gate_rows.append({"pool_seed": pool_seed, "model_seed": model_seed, "n_cand": n_cand, "pass_rate": pass_rate})
            for ratio in RATIOS:
                selected = POLICY["selection"][str(ratio)]
                params = {
                    "reactive": None,
                    "periodic": selected["periodic"],
                    "risk": selected["risk"],
                    "rul": selected["rul"],
                    "risk_gated": selected["risk"],
                    "rul_gated": selected["rul"],
                }
                for policy in POLICIES:
                    row = evaluate(eps, policy, params[policy], ratio)
                    row.update(
                        {
                            "ratio": ratio,
                            "pool_seed": pool_seed,
                            "model_seed": model_seed,
                            "policy": policy,
                            "param": "" if params[policy] is None else params[policy],
                        }
                    )
                    csv_rows.append(row)

    summary = {"by_ratio": {}, "gate_candidates": {}}
    for ratio in RATIOS:
        rrows = [r for r in csv_rows if r["ratio"] == ratio]
        by_policy = {policy: [r for r in rrows if r["policy"] == policy] for policy in POLICIES}
        summary["by_ratio"][str(ratio)] = {
            "policy": {
                policy: {
                    metric: summarize_cells(rows, metric)
                    for metric in ["cost_rate", "avail", "n_prev", "n_fail", "fails_per_1000", "false30_rate"]
                }
                for policy, rows in by_policy.items()
            },
            "contrasts": {
                "rul_minus_reactive_cost": paired_delta(by_policy, "rul", "reactive", "cost_rate"),
                "rul_minus_periodic_cost": paired_delta(by_policy, "rul", "periodic", "cost_rate"),
                "rul_gated_minus_rul_cost": paired_delta(by_policy, "rul_gated", "rul", "cost_rate"),
                "rul_gated_minus_rul_false30": paired_delta(by_policy, "rul_gated", "rul", "false30_rate"),
                "rul_gated_minus_rul_n_prev": paired_delta(by_policy, "rul_gated", "rul", "n_prev"),
                "risk_minus_rul_cost": paired_delta(by_policy, "risk", "rul", "cost_rate"),
            },
        }

    for metric in ["n_cand", "pass_rate"]:
        summary["gate_candidates"][metric] = summarize_cells(gate_rows, metric)

    result = {
        "description": "Crossed model-seed by evaluation-pool-seed audit for policy outcomes.",
        "design": "Five saved model seeds are crossed with the expanded simulator evaluation-pool seed set; the seed-matched/reused-model diagonal is compared with the full grid.",
        "note": "This is a simulator-bound stochasticity audit, not additional plant validation.",
        "summary": summary,
    }

    jdump(result, OUT + "/q1_model_seed_audit.json")
    with open(OUT + "/q1_model_seed_audit.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ratio",
            "pool_seed",
            "model_seed",
            "policy",
            "param",
            "cost_rate",
            "avail",
            "n_prev",
            "n_fail",
            "fails_per_1000",
            "wear_fails",
            "false30_rate",
            "mean_rem_at_prev",
            "op",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("wrote", OUT + "/q1_model_seed_audit.json")
    for ratio in ["5", "10", "20"]:
        pol = summary["by_ratio"][ratio]["policy"]
        print(
            "ratio",
            ratio,
            "rul all/diag",
            round(pol["rul"]["cost_rate"]["all_cells"]["mean"], 4),
            round(pol["rul"]["cost_rate"]["diagonal"]["mean"], 4),
            "gate delta cost",
            round(summary["by_ratio"][ratio]["contrasts"]["rul_gated_minus_rul_cost"]["mean"], 4),
            "gate delta fa",
            round(summary["by_ratio"][ratio]["contrasts"]["rul_gated_minus_rul_false30"]["mean"], 4),
        )


if __name__ == "__main__":
    main()

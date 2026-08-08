import csv
import json
import os
import sys

import joblib
import numpy as np
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, FEATS, jdump, load_ai4i
from simulator import C, OSF_THR, fit_params, fit_pools
from stage12_q1_robustness import aggregate, eval_policy, select_params


DD = BASE + "/data"
OUT = BASE + "/results"
CP_RATIOS = [2, 5, 10, 20]


def shap_values_positive(explainer, x):
    vals = explainer.shap_values(x)
    if isinstance(vals, list):
        return vals[1]
    if getattr(vals, "ndim", 0) == 3:
        return vals[..., 1]
    return vals


def gen_episode_warm(rng, params, pools, warm_values, max_cycles=400, shocks=True):
    ty = int(rng.choice(3, p=params["type_probs"]))
    w_twf = float(rng.uniform(params["twf_lo"], params["twf_hi"]))
    wear = float(min(rng.choice(warm_values), max(0.0, w_twf - 5.0)))
    air = params["air_mu"] + rng.normal(0, params["air_sd_stat"])
    n = len(pools["tq"])
    rows = []
    fail_mode = None
    for _ in range(max_cycles):
        air = params["air_mu"] + params["air_rho"] * (air - params["air_mu"]) + rng.normal(0, params["air_sig"])
        j = int(rng.integers(n))
        d_t = float(pools["dT"][j])
        tq = float(pools["tq"][j])
        rpm = float(pools["rpm"][j])
        proc = air + d_t
        power = tq * rpm * C
        wear += float(rng.choice(params["inc_vals"], p=params["inc_probs"]))
        rows.append((ty, air, proc, rpm, tq, wear))
        if wear >= w_twf:
            fail_mode = "TWF"
        elif wear * tq > OSF_THR[ty]:
            fail_mode = "OSF"
        elif shocks and d_t < 8.6 and rpm < 1380:
            fail_mode = "HDF"
        elif shocks and (power < 3500 or power > 9000):
            fail_mode = "PWF"
        elif shocks and rng.random() < 0.001:
            fail_mode = "RNF"
        if fail_mode:
            break
    x = np.array(rows)
    return dict(X=x, T=len(rows), fail_mode=fail_mode or "CENSOR", w_twf=w_twf)


def gen_pool(seed, n_ep):
    df = load_ai4i()
    params = fit_params(df)
    pools = fit_pools(df)
    warm_values = df["Tool wear [min]"].to_numpy(float)
    rng = np.random.default_rng(seed)
    return [gen_episode_warm(rng, params, pools, warm_values) for _ in range(n_ep)]


def add_model_outputs(eps, model_seed):
    x = np.vstack([ep["X"] for ep in eps])
    risk = joblib.load(f"{DD}/riskrf_{model_seed}.joblib").predict_proba(x)[:, 1]
    pred_rul = joblib.load(f"{DD}/rulmodel2_HGBR_{model_seed}.joblib").predict(x)
    cand = np.where((pred_rul <= 16) | (risk >= 0.5))[0]
    votes = []
    for g in range(3):
        model = joblib.load(f"{DD}/gaterf_{g}.joblib")
        vals = np.abs(shap_values_positive(shap.TreeExplainer(model), x[cand]))
        top2 = np.argsort(-vals, axis=1)[:, :2]
        votes.append(np.any(top2 == 5, axis=1))
    gate_cand = np.vstack(votes).sum(axis=0) >= 2
    gate = np.zeros(len(x), dtype=bool)
    gate[cand] = gate_cand
    offset = 0
    out = []
    for ep in eps:
        a, b = offset, offset + ep["T"]
        true_rul = np.arange(ep["T"] - 1, -1, -1, dtype=float)
        out.append(
            dict(
                T=ep["T"],
                risk=risk[a:b],
                pred_rul=pred_rul[a:b],
                true_rul=true_rul,
                gate=gate[a:b],
                wear=ep["X"][:, 5],
                torque=ep["X"][:, 4],
                wtwf=ep["w_twf"],
                mode=ep["fail_mode"],
            )
        )
        offset = b
    return out


def pool_stats(eps):
    wear = np.concatenate([ep["wear"] for ep in eps])
    modes = [ep["mode"] for ep in eps]
    cycles = sum(ep["T"] for ep in eps)
    return dict(
        n_episodes=len(eps),
        n_cycles=int(cycles),
        mean_episode_length=float(np.mean([ep["T"] for ep in eps])),
        wear_mean=float(np.mean(wear)),
        wear_sd=float(np.std(wear)),
        mode_rates_per_1000_cycles={m: float(1000 * modes.count(m) / cycles) for m in ["TWF", "OSF", "HDF", "PWF", "RNF", "CENSOR"]},
    )


def main():
    validation = add_model_outputs(gen_pool(17900, 300), 0)
    pools = [add_model_outputs(gen_pool(17000 + seed, 400), seed) for seed in range(5)]
    rows = []
    result = {
        "description": "Warm-start high-wear simulator sensitivity using AI4I tool-wear values as initial wear.",
        "validation_pool": pool_stats(validation),
        "evaluation_pools": [pool_stats(p) for p in pools],
        "selection": {},
        "by_ratio": {},
        "note": "This is a stress test of high-wear exposure, not a replacement for plant validation.",
    }
    for ratio in CP_RATIOS:
        selected = select_params(validation, ratio)
        # Preserve the original RUL horizons for comparability with the main run.
        selected["rul"] = 8 if ratio == 2 else 16
        result["selection"][str(ratio)] = selected
        variants = [
            ("rul_base", "rul", selected["rul"]),
            ("shap_gate", "rul_gated", selected["rul"]),
            ("strict_rul_selected", "strict_rul", selected["strict_rul"]),
            ("raw_wear_selected", "raw_wear", selected["raw_wear"]),
            ("wear_torque_selected", "wear_torque", selected["wear_torque"]),
            ("periodic_fine_selected", "periodic_fine", selected["periodic_fine"]),
            ("oracle_wear_final", "oracle_wear_final", None),
        ]
        by_variant = {}
        for name, policy_name, param in variants:
            vals = [eval_policy(eps, policy_name, param, ratio) for eps in pools]
            by_variant[name] = vals
            for seed, row in enumerate(vals):
                rows.append({"ratio": ratio, "seed": seed, "variant": name, **row})
        result["by_ratio"][str(ratio)] = {
            name: aggregate(vals) for name, vals in by_variant.items()
        }
    jdump(result, OUT + "/q1_high_wear_sensitivity.json")
    with open(OUT + "/q1_high_wear_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ratio",
            "seed",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT + "/q1_high_wear_sensitivity.json")
    print(json.dumps(result["validation_pool"], indent=1))


if __name__ == "__main__":
    main()

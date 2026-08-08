import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, jdump


DD = BASE + "/data"
OUT = BASE + "/results"
SP = json.load(open(OUT + "/sim_params.json"))
POLICY = json.load(open(OUT + "/policy_eval.json"))
EVAL_SEEDS = list(POLICY.get("eval_seeds", range(5)))
MEAN_INC = float(np.dot(SP["inc_vals"], SP["inc_probs"]))
CP, CD_RATE, DP, DC = 1.0, 0.05, 2, 10
RATIOS = [2, 5, 10, 20]
FALSE_THRESHOLDS = [10, 20, 30, 40]


def load_pool(seed):
    z = np.load(f"{DD}/pool2_pe_{seed}.npz", allow_pickle=True)
    g = np.load(f"{OUT}/gate_pe_{seed}.npz", allow_pickle=True)
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
                rul=g["rul"][a:b],
                gate=gate_full[a:b],
                wear=z["X"][a:b, 5],
                wtwf=float(z["wtwf"][e]),
                mode=str(z["modes"][e]),
            )
        )
    return eps


def terminal_outcome(ep):
    return dict(prev=0, fail=1, op=ep["T"], down=DC, mode=ep["mode"], rem=None)


def preventive_outcome(ep, t):
    rem = (ep["wtwf"] - ep["wear"][t]) / MEAN_INC
    return dict(prev=1, fail=0, op=t + 1, down=DP, mode=None, rem=float(rem))


def trigger_index(ep, policy, param, gate_mask=None):
    T = ep["T"]
    if policy == "reactive":
        return None
    if policy == "periodic":
        return param - 1 if param - 1 <= T - 2 else None
    if policy == "risk":
        trig = ep["risk"] >= param
    elif policy == "rul":
        trig = ep["rul"] <= param
    else:
        raise ValueError(policy)
    if gate_mask is not None:
        trig = trig & gate_mask
    idx = np.where(trig[: T - 1])[0]
    return int(idx[0]) if len(idx) else None


def outcomes(eps, policy, param, gate_kind=None, rng=None, keep_count=None):
    raw = []
    for ep in eps:
        gate_mask = None
        if gate_kind == "shap":
            gate_mask = ep["gate"]
        elif gate_kind == "strict_rul":
            gate_mask = ep["rul"] <= max(3, param // 2)
        elif gate_kind == "oracle_false30":
            rem_cycle = (ep["wtwf"] - ep["wear"]) / MEAN_INC
            gate_mask = rem_cycle <= 30
        t = trigger_index(ep, policy, param, gate_mask)
        raw.append(terminal_outcome(ep) if t is None else preventive_outcome(ep, t))

    if gate_kind == "random_action_matched":
        action_idx = [i for i, o in enumerate(raw) if o["prev"]]
        keep_count = min(int(keep_count), len(action_idx))
        keep = set(rng.choice(action_idx, size=keep_count, replace=False).tolist()) if keep_count else set()
        matched = []
        for i, o in enumerate(raw):
            if o["prev"] and i not in keep:
                matched.append(terminal_outcome(eps[i]))
            else:
                matched.append(o)
        return matched
    return raw


def summarize(out, ratio, false_threshold=30, extra_burden=0.0):
    op = sum(o["op"] for o in out)
    down = sum(o["down"] for o in out)
    np_ = sum(o["prev"] for o in out)
    nf = sum(o["fail"] for o in out)
    cost = np_ * (CP + extra_burden) + nf * ratio * CP + CD_RATE * down
    false_n = sum(1 for o in out if o["prev"] and o["rem"] is not None and o["rem"] > false_threshold)
    rems = [o["rem"] for o in out if o["prev"] and o["rem"] is not None]
    return dict(
        cost_rate=float(cost / op),
        avail=float(op / (op + down)),
        n_prev=float(np_),
        n_fail=float(nf),
        fails_per_1000=float(1000 * nf / op),
        false_rate=float(false_n / np_) if np_ else 0.0,
        false_n=float(false_n),
        mean_rem=float(np.mean(rems)) if rems else 0.0,
        op=float(op),
        total_cost=float(cost),
    )


def mean_sd(vals):
    return dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))


def aggregate(rows, metrics):
    return {m: mean_sd([r[m] for r in rows]) for m in metrics}


def main():
    policy = POLICY
    pools = [load_pool(s) for s in EVAL_SEEDS]
    summary = {
        "description": "Q1 audit sensitivity and control analyses derived from the saved policy evaluation pools.",
        "false_thresholds": FALSE_THRESHOLDS,
        "controls": {},
        "false_threshold_sensitivity": {},
        "break_even_burden": {},
        "parameter_neighbors": {},
        "scenario_note": "These checks use the existing calibrated simulator pools; they do not convert simulator evidence into plant evidence.",
    }
    csv_rows = []

    for ratio in RATIOS:
        rkey = str(ratio)
        h = int(policy["selection"][rkey]["rul"])
        k = int(policy["selection"][rkey]["periodic"])
        base_rows, shap_rows, strict_rows, oracle_rows, random_rows = [], [], [], [], []
        for seed, eps in zip(EVAL_SEEDS, pools):
            rng = np.random.default_rng(90000 + 100 * ratio + seed)
            base = outcomes(eps, "rul", h)
            shap = outcomes(eps, "rul", h, gate_kind="shap")
            strict = outcomes(eps, "rul", h, gate_kind="strict_rul")
            oracle = outcomes(eps, "rul", h, gate_kind="oracle_false30")
            target_count = sum(o["prev"] for o in shap)
            random_matched = outcomes(
                eps,
                "rul",
                h,
                gate_kind="random_action_matched",
                rng=rng,
                keep_count=target_count,
            )
            for name, outs, coll in [
                ("rul_base", base, base_rows),
                ("shap_gate", shap, shap_rows),
                ("strict_rul_gate", strict, strict_rows),
                ("oracle_false30_gate", oracle, oracle_rows),
                ("random_action_matched", random_matched, random_rows),
            ]:
                row = summarize(outs, ratio, false_threshold=30)
                coll.append(row)
                csv_rows.append({"ratio": ratio, "seed": seed, "analysis": "gate_control", "variant": name, **row})

        metrics = ["cost_rate", "avail", "n_prev", "n_fail", "fails_per_1000", "false_rate", "mean_rem"]
        summary["controls"][rkey] = {
            "rul_base": aggregate(base_rows, metrics),
            "shap_gate": aggregate(shap_rows, metrics),
            "strict_rul_gate": aggregate(strict_rows, metrics),
            "oracle_false30_gate": aggregate(oracle_rows, metrics),
            "random_action_matched": aggregate(random_rows, metrics),
        }

        threshold_block = {}
        for thr in FALSE_THRESHOLDS:
            base_thr, shap_thr = [], []
            for eps in pools:
                base_thr.append(summarize(outcomes(eps, "rul", h), ratio, false_threshold=thr))
                shap_thr.append(summarize(outcomes(eps, "rul", h, gate_kind="shap"), ratio, false_threshold=thr))
            threshold_block[str(thr)] = {
                "rul_base_false_rate": mean_sd([r["false_rate"] for r in base_thr]),
                "shap_gate_false_rate": mean_sd([r["false_rate"] for r in shap_thr]),
            }
        summary["false_threshold_sensitivity"][rkey] = threshold_block

        be = []
        for eps in pools:
            p_out = outcomes(eps, "periodic", k)
            r_out = outcomes(eps, "rul", h)
            p0 = summarize(p_out, ratio, false_threshold=30)
            r0 = summarize(r_out, ratio, false_threshold=30)
            denom = p0["n_prev"] - r0["n_prev"]
            burden = (r0["total_cost"] - p0["total_cost"]) / denom if denom > 0 else float("nan")
            be.append(float(burden))
        summary["break_even_burden"][rkey] = mean_sd(be)

        neighbors = sorted(set([max(1, h // 2), h, h + 4, h + 8]))
        nb = {}
        for hv in neighbors:
            rows = [summarize(outcomes(eps, "rul", hv), ratio, false_threshold=30) for eps in pools]
            nb[str(hv)] = aggregate(rows, ["cost_rate", "n_prev", "false_rate", "fails_per_1000"])
        summary["parameter_neighbors"][rkey] = nb

    jdump(summary, OUT + "/q1_sensitivity.json")
    with open(OUT + "/q1_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
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
            "false_n",
            "mean_rem",
            "op",
            "total_cost",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)
    print(json.dumps(summary["controls"], indent=2))
    print("wrote", OUT + "/q1_sensitivity.json")


if __name__ == "__main__":
    main()

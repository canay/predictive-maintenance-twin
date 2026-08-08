import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#F0E442"]
MODEL_LABELS = {
    "constant": "Constant",
    "ridge": "Ridge",
    "hgb": "HGBR",
    "rf": "RF",
}


def find_base_dir():
    env_base = os.environ.get("FH8_BASE")
    if env_base:
        return Path(env_base).resolve()
    return Path(__file__).resolve().parents[1]


def save(fig, fig_dir, name):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("FH8_RUN_ID", "cmapss_external_local"))
    args = parser.parse_args()

    base = find_base_dir()
    results_dir = base / "results" / args.run_id
    fig_dir = base / "figures" / args.run_id
    bench_path = results_dir / "benchmark_results.csv"
    metric_path = results_dir / "metric_summary.csv"
    if not bench_path.exists():
        raise FileNotFoundError(bench_path)

    bench = pd.read_csv(bench_path)
    metric = pd.read_csv(metric_path) if metric_path.exists() else (
        bench.groupby(["dataset", "model"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
             nasa_s_score_mean=("nasa_s_score", "mean"),
             overestimate_fraction_mean=("overestimate_fraction", "mean"),
             total_cell_seconds_mean=("total_cell_seconds", "mean"))
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    datasets = sorted(metric["dataset"].unique())
    models = [m for m in ["constant", "ridge", "hgb", "rf"] if m in set(metric["model"])]
    x = np.arange(len(datasets))
    width = 0.18 if len(models) >= 4 else 0.24

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for i, model in enumerate(models):
        sub = metric[metric["model"] == model].set_index("dataset").reindex(datasets)
        offset = (i - (len(models) - 1) / 2) * width
        axes[0].bar(x + offset, sub["rmse_mean"], width, yerr=sub.get("rmse_sd"), color=OI[i], capsize=2, label=MODEL_LABELS.get(model, model))
        axes[1].bar(x + offset, sub["overestimate_fraction_mean"], width, color=OI[i], label=MODEL_LABELS.get(model, model))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("RMSE [cycles]")
    axes[0].set_title("External RUL accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Over-estimation fraction")
    axes[1].set_title("Safety-relevant bias")
    axes[0].legend(frameon=False, ncol=2)
    save(fig, fig_dir, "fig_cmapss_external_metrics")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for i, model in enumerate(models):
        sub = metric[metric["model"] == model]
        ax.scatter(sub["total_cell_seconds_mean"], sub["rmse_mean"], s=45, color=OI[i], label=MODEL_LABELS.get(model, model))
        for _, row in sub.iterrows():
            ax.text(row["total_cell_seconds_mean"], row["rmse_mean"], row["dataset"], fontsize=7, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Train+predict time per cell [s, log scale]")
    ax.set_ylabel("RMSE [cycles]")
    ax.set_title("Runtime-quality trade-off")
    ax.legend(frameon=False, ncol=2)
    save(fig, fig_dir, "fig_cmapss_runtime_tradeoff")

    pred_dir = results_dir / "predictions"
    best = metric.loc[metric.groupby("dataset")["rmse_mean"].idxmin()]
    n = len(best)
    fig, axes = plt.subplots(1, n, figsize=(max(3.0 * n, 5.5), 3.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, best.iterrows()):
        ds = row["dataset"]
        model = row["model"]
        pred_files = sorted(pred_dir.glob(f"{ds}_{model}_seed*.csv"))
        if not pred_files:
            continue
        pred = pd.read_csv(pred_files[0])
        ax.scatter(pred["true_rul"], pred["pred_rul"], s=12, alpha=0.7, color=OI[0])
        lim = max(float(pred["true_rul"].max()), float(pred["pred_rul"].max()), 1.0)
        ax.plot([0, lim], [0, lim], color="#444444", lw=1.0, linestyle="--")
        ax.set_title(f"{ds}: {MODEL_LABELS.get(model, model)}")
        ax.set_xlabel("True RUL")
        ax.set_xlim(0, lim * 1.05)
        ax.set_ylim(0, lim * 1.05)
    axes[0].set_ylabel("Predicted RUL")
    save(fig, fig_dir, "fig_cmapss_prediction_scatter")

    summary = {
        "run_id": args.run_id,
        "figure_dir": str(fig_dir),
        "figures": [
            "fig_cmapss_external_metrics",
            "fig_cmapss_runtime_tradeoff",
            "fig_cmapss_prediction_scatter",
        ],
        "best_by_dataset": best.to_dict(orient="records"),
    }
    with (fig_dir / "figure_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NASA_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
DATASETS = ("FD001", "FD002", "FD003", "FD004")
BASE_COLS = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
RAW_COLS = ["unit", "cycle"] + BASE_COLS


def iso_now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def find_base_dir():
    env_base = os.environ.get("FH8_BASE")
    if env_base:
        return Path(env_base).resolve()
    return Path(__file__).resolve().parents[1]


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=float) + "\n")


def append_csv(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resource_snapshot():
    snap = {
        "timestamp": iso_now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "loadavg": None,
        "rss_mb": None,
        "system_memory_total_mb": None,
        "system_memory_available_mb": None,
    }
    try:
        snap["loadavg"] = list(os.getloadavg())
    except Exception:
        pass
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem = psutil.virtual_memory()
        snap["rss_mb"] = proc.memory_info().rss / (1024 ** 2)
        snap["system_memory_total_mb"] = mem.total / (1024 ** 2)
        snap["system_memory_available_mb"] = mem.available / (1024 ** 2)
        snap["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except Exception:
        pass
    return snap


def download_and_extract(data_dir, timings_path):
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "CMAPSSData.zip"
    extract_dir = data_dir / "CMAPSSData"
    required = extract_dir / "train_FD001.txt"
    if required.exists():
        return {
            "source_url": NASA_CMAPSS_URL,
            "zip_path": str(zip_path),
            "extract_dir": str(extract_dir),
            "downloaded": False,
            "zip_md5": md5(zip_path) if zip_path.exists() else None,
        }

    t0 = time.perf_counter()
    if not zip_path.exists():
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(NASA_CMAPSS_URL, timeout=60) as r, zip_path.open("wb") as f:
                    shutil.copyfileobj(r, f)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * attempt)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    elapsed = time.perf_counter() - t0
    append_jsonl(timings_path, {
        "stage": "download_extract",
        "seconds": elapsed,
        "source_url": NASA_CMAPSS_URL,
        "zip_bytes": zip_path.stat().st_size,
        "zip_md5": md5(zip_path),
        "resource": resource_snapshot(),
    })
    return {
        "source_url": NASA_CMAPSS_URL,
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "downloaded": True,
        "zip_md5": md5(zip_path),
    }


def cmapss_root(data_dir):
    root = data_dir / "CMAPSSData"
    if (root / "train_FD001.txt").exists():
        return root
    nested = root / "CMAPSSData"
    if (nested / "train_FD001.txt").exists():
        return nested
    raise FileNotFoundError(f"Could not locate train_FD001.txt under {root}")


def read_raw(root, ds, split):
    path = root / f"{split}_{ds}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLS)


def read_rul(root, ds):
    path = root / f"RUL_{ds}.txt"
    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].astype(float).to_numpy()
    return values


def add_engine_rul(df, cap):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    out = df.copy()
    out["rul_raw"] = max_cycle - out["cycle"]
    out["rul_capped"] = np.minimum(out["rul_raw"], cap)
    out["observed_life"] = max_cycle
    return out


def engineer_features(df):
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True).copy()
    frames = [
        df[["unit", "cycle"]],
        df[["cycle"]].rename(columns={"cycle": "age_cycles"}),
        df[BASE_COLS].add_prefix("last_"),
    ]
    grouped = df.groupby("unit", sort=False)[BASE_COLS]
    for w in (5, 30):
        mean = grouped.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        std = grouped.rolling(w, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        delta = grouped.diff(w - 1).fillna(0.0)
        frames.append(mean.add_prefix(f"mean{w}_"))
        frames.append(std.add_prefix(f"std{w}_"))
        frames.append(delta.add_prefix(f"delta{w}_"))
    feats = pd.concat(frames, axis=1)
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prepare_dataset(root, ds, cap, stride, max_train_engines=None, max_test_engines=None):
    train_raw = add_engine_rul(read_raw(root, ds, "train"), cap)
    test_raw = read_raw(root, ds, "test")

    if max_train_engines:
        train_units = sorted(train_raw["unit"].unique())[:max_train_engines]
        train_raw = train_raw[train_raw["unit"].isin(train_units)].copy()
    if max_test_engines:
        test_units = sorted(test_raw["unit"].unique())[:max_test_engines]
        test_raw = test_raw[test_raw["unit"].isin(test_units)].copy()

    test_max = test_raw.groupby("unit")["cycle"].transform("max")
    test_raw = test_raw.copy()
    test_raw["observed_life"] = test_max

    train_feats = engineer_features(train_raw)
    keep = (
        (train_raw["cycle"] % stride == 0)
        | (train_raw["rul_raw"] <= 50)
        | (train_raw["cycle"] == train_raw["observed_life"])
    )
    train_feats = train_feats.loc[keep].reset_index(drop=True)
    y_train = train_raw.loc[keep, "rul_capped"].astype(float).to_numpy()

    test_feats_all = engineer_features(test_raw)
    last_idx = test_raw.groupby("unit")["cycle"].idxmax().to_numpy()
    test_feats = test_feats_all.loc[last_idx].sort_values("unit").reset_index(drop=True)
    y_test_full = read_rul(root, ds)
    if max_test_engines:
        y_test_full = y_test_full[:max_test_engines]
    y_test = y_test_full.astype(float)

    drop_cols = ["unit", "cycle"]
    x_train = train_feats.drop(columns=drop_cols).to_numpy(dtype=np.float32)
    x_test = test_feats.drop(columns=drop_cols).to_numpy(dtype=np.float32)
    feature_names = [c for c in train_feats.columns if c not in drop_cols]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "train_rows_raw": int(len(train_raw)),
        "train_rows_used": int(len(y_train)),
        "test_rows_raw": int(len(test_raw)),
        "test_engines": int(len(y_test)),
        "train_engines": int(train_raw["unit"].nunique()),
    }


class ConstantRegressor:
    def __init__(self):
        self.value_ = None

    def fit(self, x, y):
        self.value_ = float(np.median(y))
        return self

    def predict(self, x):
        return np.full(x.shape[0], self.value_, dtype=float)


def make_model(name, seed, n_jobs):
    if name == "constant":
        return ConstantRegressor()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed))
    if name == "hgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            early_stopping=True,
            random_state=seed,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=180,
            max_features=0.55,
            min_samples_leaf=3,
            n_jobs=n_jobs,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def nasa_s_score(y_true, y_pred):
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(score))


def score_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nasa_s_score": nasa_s_score(y_true, y_pred),
        "bias_mean": float(np.mean(err)),
        "overestimate_fraction": float(np.mean(err > 0)),
        "underestimate_fraction": float(np.mean(err < 0)),
        "within_10_cycles": float(np.mean(np.abs(err) <= 10)),
        "within_20_cycles": float(np.mean(np.abs(err) <= 20)),
        "within_30pct": float(np.mean(np.abs(err) <= np.maximum(0.3 * y_true, 3.0))),
        "p95_abs_error": float(np.percentile(np.abs(err), 95)),
        "max_overestimate": float(np.max(err)),
        "max_underestimate": float(np.min(err)),
    }


def existing_keys(path):
    path = Path(path)
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(zip(df["dataset"].astype(str), df["model"].astype(str), df["seed"].astype(int)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("FH8_RUN_ID", "cmapss_external_local"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=["constant", "ridge", "hgb", "rf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rul-cap", type=float, default=125.0)
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("FH8_N_JOBS", "2")))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-engines", type=int, default=None)
    parser.add_argument("--max-test-engines", type=int, default=None)
    args = parser.parse_args()

    if args.smoke:
        args.datasets = [args.datasets[0]]
        args.models = ["constant", "ridge"]
        args.seeds = [0]
        args.max_train_engines = args.max_train_engines or 20
        args.max_test_engines = args.max_test_engines or 20

    base = find_base_dir()
    out_dir = base / "results" / args.run_id
    pred_dir = out_dir / "predictions"
    data_dir = base / "data" / "cmapss"
    logs_dir = base / "logs"
    for d in (out_dir, pred_dir, data_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    timings_path = out_dir / "timings.jsonl"
    resource_path = out_dir / "resource_report.json"
    bench_path = out_dir / "benchmark_results.csv"
    fieldnames = [
        "run_id", "dataset", "model", "seed", "start_time", "end_time",
        "train_seconds", "predict_seconds", "total_cell_seconds",
        "train_rows_raw", "train_rows_used", "train_engines", "test_rows_raw",
        "test_engines", "feature_count", "rmse", "mae", "nasa_s_score",
        "bias_mean", "overestimate_fraction", "underestimate_fraction",
        "within_10_cycles", "within_20_cycles", "within_30pct",
        "p95_abs_error", "max_overestimate", "max_underestimate",
    ]

    run_meta = {
        "run_id": args.run_id,
        "base_dir": str(base),
        "created_at": iso_now(),
        "datasets": args.datasets,
        "models": args.models,
        "seeds": args.seeds,
        "rul_cap": args.rul_cap,
        "n_jobs": args.n_jobs,
        "smoke": args.smoke,
        "source": {
            "name": "NASA C-MAPSS / Turbofan Engine Degradation Simulation Data Set",
            "url": NASA_CMAPSS_URL,
            "scope": "External temporal run-to-failure prognostics validation",
        },
        "resource_start": resource_snapshot(),
    }
    atomic_json(run_meta, out_dir / "run_manifest.json")

    data_meta = download_and_extract(data_dir, timings_path)
    root = cmapss_root(data_dir)
    atomic_json(data_meta, out_dir / "data_manifest.json")

    done = existing_keys(bench_path)
    dataset_cache = {}
    dataset_summaries = []

    for ds in args.datasets:
        stride = 1 if ds in ("FD001", "FD003") else 2
        t0 = time.perf_counter()
        prepared = prepare_dataset(
            root, ds, args.rul_cap, stride,
            max_train_engines=args.max_train_engines,
            max_test_engines=args.max_test_engines,
        )
        elapsed = time.perf_counter() - t0
        dataset_cache[ds] = prepared
        ds_summary = {
            "dataset": ds,
            "stride": stride,
            "train_rows_raw": prepared["train_rows_raw"],
            "train_rows_used": prepared["train_rows_used"],
            "train_engines": prepared["train_engines"],
            "test_rows_raw": prepared["test_rows_raw"],
            "test_engines": prepared["test_engines"],
            "feature_count": len(prepared["feature_names"]),
            "feature_seconds": elapsed,
        }
        dataset_summaries.append(ds_summary)
        append_jsonl(timings_path, {
            "stage": "feature_engineering",
            "dataset": ds,
            "seconds": elapsed,
            "summary": ds_summary,
            "resource": resource_snapshot(),
        })

        for model_name in args.models:
            for seed in args.seeds:
                key = (ds, model_name, int(seed))
                if key in done:
                    continue
                start = iso_now()
                cell_t0 = time.perf_counter()
                model = make_model(model_name, seed, args.n_jobs)
                train_t0 = time.perf_counter()
                model.fit(prepared["x_train"], prepared["y_train"])
                train_seconds = time.perf_counter() - train_t0
                pred_t0 = time.perf_counter()
                pred = model.predict(prepared["x_test"])
                pred = np.clip(pred, 0.0, args.rul_cap)
                predict_seconds = time.perf_counter() - pred_t0
                metrics = score_predictions(prepared["y_test"], pred)
                total_cell_seconds = time.perf_counter() - cell_t0

                pred_df = pd.DataFrame({
                    "unit": np.arange(1, len(pred) + 1),
                    "true_rul": prepared["y_test"],
                    "pred_rul": pred,
                    "error": pred - prepared["y_test"],
                })
                pred_path = pred_dir / f"{ds}_{model_name}_seed{seed}.csv"
                pred_df.to_csv(pred_path, index=False)

                row = {
                    "run_id": args.run_id,
                    "dataset": ds,
                    "model": model_name,
                    "seed": int(seed),
                    "start_time": start,
                    "end_time": iso_now(),
                    "train_seconds": train_seconds,
                    "predict_seconds": predict_seconds,
                    "total_cell_seconds": total_cell_seconds,
                    "train_rows_raw": prepared["train_rows_raw"],
                    "train_rows_used": prepared["train_rows_used"],
                    "train_engines": prepared["train_engines"],
                    "test_rows_raw": prepared["test_rows_raw"],
                    "test_engines": prepared["test_engines"],
                    "feature_count": len(prepared["feature_names"]),
                    **metrics,
                }
                append_csv(bench_path, row, fieldnames)
                append_jsonl(timings_path, {
                    "stage": "model_cell",
                    "dataset": ds,
                    "model": model_name,
                    "seed": int(seed),
                    "train_seconds": train_seconds,
                    "predict_seconds": predict_seconds,
                    "total_cell_seconds": total_cell_seconds,
                    "metrics": metrics,
                    "resource": resource_snapshot(),
                })
                done.add(key)
                print(
                    f"{ds} {model_name} seed={seed} "
                    f"rmse={metrics['rmse']:.3f} mae={metrics['mae']:.3f} "
                    f"s={metrics['nasa_s_score']:.1f} "
                    f"time={total_cell_seconds:.1f}s",
                    flush=True,
                )

    ds_df = pd.DataFrame(dataset_summaries)
    ds_df.to_csv(out_dir / "dataset_summary.csv", index=False)

    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        metric_summary = (
            bench.groupby(["dataset", "model"], as_index=False)
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_sd=("rmse", "std"),
                mae_mean=("mae", "mean"),
                mae_sd=("mae", "std"),
                nasa_s_score_mean=("nasa_s_score", "mean"),
                nasa_s_score_sd=("nasa_s_score", "std"),
                overestimate_fraction_mean=("overestimate_fraction", "mean"),
                within_20_cycles_mean=("within_20_cycles", "mean"),
                train_seconds_mean=("train_seconds", "mean"),
                predict_seconds_mean=("predict_seconds", "mean"),
                total_cell_seconds_mean=("total_cell_seconds", "mean"),
            )
            .sort_values(["dataset", "rmse_mean"])
        )
        metric_summary.to_csv(out_dir / "metric_summary.csv", index=False)
        best = metric_summary.loc[metric_summary.groupby("dataset")["rmse_mean"].idxmin()].copy()
        best.to_csv(out_dir / "best_by_dataset.csv", index=False)
        summary = {
            "run_id": args.run_id,
            "completed_at": iso_now(),
            "n_rows": int(len(bench)),
            "expected_rows": int(len(args.datasets) * len(args.models) * len(args.seeds)),
            "datasets": dataset_summaries,
            "best_by_dataset": best.to_dict(orient="records"),
            "resource_end": resource_snapshot(),
        }
        atomic_json(summary, out_dir / "summary.json")

    final_resource = {
        "run_id": args.run_id,
        "created_at": run_meta["created_at"],
        "completed_at": iso_now(),
        "start": run_meta["resource_start"],
        "end": resource_snapshot(),
        "environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "FH8_N_JOBS": os.environ.get("FH8_N_JOBS"),
        },
    }
    atomic_json(final_resource, resource_path)
    print(f"completed run_id={args.run_id} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()

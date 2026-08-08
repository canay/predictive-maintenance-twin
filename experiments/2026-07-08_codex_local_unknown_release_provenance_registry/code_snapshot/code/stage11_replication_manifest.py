import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, jdump


ROOT = Path(BASE)


def sha256(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row(artifact, script, command, inputs, outputs, seeds, notes):
    return {
        "artifact": artifact,
        "script": script,
        "command": command,
        "inputs": inputs,
        "outputs": outputs,
        "seeds": seeds,
        "notes": notes,
    }


def file_info(relpath):
    path = ROOT / relpath
    return {
        "path": relpath,
        "exists": path.is_file(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def main():
    rows = [
        row(
            "AI4I risk table and fig_risk",
            "code/stage0_data.py; code/stage1_risk.py; code/stage7_figs.py",
            "python code/stage0_data.py; python code/stage1_risk.py; python code/stage7_figs.py",
            "data/ai4i2020.csv",
            "results/risk/*.json; manuscript/fig_risk.png; figures/fig_risk.pdf source/vector export",
            "0,1,2,3,4",
            "LR, RF, and HGB 70/30 stratified splits.",
        ),
        row(
            "Simulator calibration table and fig_calibration",
            "code/stage2_calib.py; code/stage7_figs.py",
            "python code/stage2_calib.py; python code/stage7_figs.py",
            "data/ai4i2020.csv",
            "results/sim_params.json; results/calibration.json; manuscript/fig_calibration.png; figures/fig_calibration.pdf source/vector export",
            "12345",
            "600-episode calibration pool.",
        ),
        row(
            "RUL prognostic table and fig_rul",
            "code/stage3_rul.py; code/stage7_figs.py; code/stage9b_stats.py",
            "python code/stage3_rul.py; python code/stage7_figs.py; python code/stage9b_stats.py",
            "data/pool2_dtr_*.npz; data/pool2_dte_*.npz",
            "results/rul/*.json; results/rul_allmodes/*.json; manuscript/fig_rul.png; figures/fig_rul.pdf source/vector export; results/stats_extra.json",
            "0,1,2,3,4",
            "Episode-level split; HGBR and MLP.",
        ),
        row(
            "Policy and explanation-gate tables/figures",
            "code/stage4a_models.py; code/stage4b_gate.py; code/stage4c_policy.py; code/stage7_figs.py; code/stage9b_stats.py",
            "python code/stage4a_models.py; python code/stage4b_gate.py; python code/stage4c_policy.py; python code/stage7_figs.py; python code/stage9b_stats.py",
            "data/riskrf_*.joblib; data/rulmodel2_HGBR_*.joblib; data/pool2_p*.npz",
            "results/gate_*.npz; results/policy_eval.json; results/stats_extra.json; manuscript/fig_policy.png; manuscript/fig_ablation.png; figures/fig_policy.pdf and figures/fig_ablation.pdf source/vector exports",
            "validation pool 7900; evaluation pools 7000-7014; saved model seeds 0-4 reused cyclically; gate RF seeds 100-102",
            "Policy parameter selection and evaluation are separated by pool; seed expansion uses additional simulator pools without retraining risk/RUL models.",
        ),
        row(
            "Q1 sensitivity and gate-control analysis",
            "code/stage10_q1_sensitivity.py",
            "python code/stage10_q1_sensitivity.py",
            "results/policy_eval.json; results/gate_pe_*.npz; data/pool2_pe_*.npz",
            "results/q1_sensitivity.json; results/q1_sensitivity.csv",
            "7000-7014; random thinning seeds 90000 + 100*ratio + seed",
            "Derived controls; no new plant evidence.",
        ),
        row(
            "Q1 robustness comparator and candidate-gate validation",
            "code/stage12_q1_robustness.py",
            "python code/stage12_q1_robustness.py",
            "results/policy_eval.json; results/gate_pe_*.npz; data/pool2_pe_*.npz; C-MAPSS prediction CSV files",
            "results/q1_robustness.json; results/q1_robustness.csv",
            "evaluation pools 7000-7014; bootstrap seeds fixed inside the script",
            "Adds raw-wear, wear-torque, fine periodic, strict-RUL, oracle, candidate-level gate metrics, and C-MAPSS paired context.",
        ),
        row(
            "High-wear warm-start sensitivity",
            "code/stage13_high_wear_sensitivity.py",
            ".venv-sklearn172/Scripts/python.exe code/stage13_high_wear_sensitivity.py",
            "data/ai4i2020.csv; data/riskrf_*.joblib; data/rulmodel2_HGBR_*.joblib; data/gaterf_*.joblib",
            "results/q1_high_wear_sensitivity.json; results/q1_high_wear_sensitivity.csv",
            "validation pool 17900; evaluation pools 17000-17004; saved model seeds 0-4; gate RF seeds 100-102",
            "Warm-starts initial tool wear from the empirical AI4I tool-wear distribution; run in a scikit-learn 1.7.2 venv to match saved model artifacts.",
        ),
        row(
            "Model-seed by evaluation-pool-seed audit",
            "code/stage14_model_seed_audit.py",
            ".venv-sklearn172/Scripts/python.exe code/stage14_model_seed_audit.py",
            "data/pool2_pe_*.npz; data/riskrf_*.joblib; data/rulmodel2_HGBR_*.joblib; data/gaterf_*.joblib; results/policy_eval.json",
            "results/q1_model_seed_audit.json; results/q1_model_seed_audit.csv",
            "evaluation pools 7000-7014 crossed with saved model seeds 0-4; gate RF seeds 100-102",
            "Crosses model and evaluation-pool seeds to check whether the deployed cyclic model/pool mapping drives the policy conclusions.",
        ),
        row(
            "Attribution fidelity table and fig_attribution",
            "code/stage5_shap.py; code/stage5c_pos.py; code/stage5b_agg.py; code/stage7_figs.py",
            "python code/stage5_shap.py; python code/stage5c_pos.py; python code/stage5b_agg.py; python code/stage7_figs.py",
            "data/ai4i2020.csv",
            "results/shap/*.json; results/shap_summary.json; manuscript/fig_attribution.png; figures/fig_attribution.pdf source/vector export",
            "0,1,2,3,4",
            "TreeSHAP audit for per-mode RF classifiers.",
        ),
        row(
            "External C-MAPSS validation",
            "code/run_cmapss_external_validation.py; code/plot_cmapss_external_validation.py",
            "python code/run_cmapss_external_validation.py --run-id cmapss_external_20260614_121237; python code/plot_cmapss_external_validation.py --run-id cmapss_external_20260614_121237",
            "data/cmapss/CMAPSSData.zip or NASA URL; official train/test/RUL files",
            "results/cmapss_external_vps_20260614_121237/*.csv/json/jsonl; manuscript/fig_cmapss_external_metrics.png; figures/fig_cmapss_external_metrics.pdf and figures/cmapss_external_vps_20260614_121237/ source exports",
            "0,1,2 for HGBR/RF; deterministic baselines repeated",
            "Run was executed on a Linux VPS; local folder preserves manifests and metrics.",
        ),
        row(
            "Manuscript numeric consistency check",
            "code/verify_tex.py",
            "python code/verify_tex.py",
            "manuscript/SCI-f08-predictive_maintenance_twin.tex; results/*.json",
            "terminal validation log",
            "not applicable",
            "Checks manuscript strings against saved result artifacts.",
        ),
    ]
    manifest = {
        "project_short_name": "predictive_maintenance_twin",
        "planned_repository": "https://github.com/canay/predictive-maintenance-twin",
        "repository_status": "local reproducibility package complete; GitHub repository creation/finalization, commit hash, and link test are final submission-time actions; no separate archival DOI is planned unless requested by the journal",
        "single_author": True,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "items": rows,
        "tex_facing_figures": [
            file_info("manuscript/fig_architecture.png"),
            file_info("manuscript/fig_risk.png"),
            file_info("manuscript/fig_calibration.png"),
            file_info("manuscript/fig_rul.png"),
            file_info("manuscript/fig_cmapss_external_metrics.png"),
            file_info("manuscript/fig_policy.png"),
            file_info("manuscript/fig_ablation.png"),
            file_info("manuscript/fig_attribution.png"),
        ],
        "figure_source_output_layout": {
            "tex_facing_outputs": "manuscript/fig_* files are the Editorial Manager / TeX-facing figure outputs kept beside the manuscript TeX.",
            "source_vector_exports": "figures/fig_*.pdf and figures/cmapss_external_vps_20260614_121237/ are source/export assets kept near, but separate from, the TeX-facing outputs.",
            "archived_old_png_duplicates": "archive/2026-07-08_release_provenance_cleanup/stale_root_figures/ and archive/2026-07-08_release_provenance_cleanup/final_png_duplicates_in_figures/",
            "archived_source_pdf_duplicates": "archive/2026-07-08_release_provenance_cleanup/source_pdf_duplicates_in_manuscript/",
            "cmapss_smoke_figure_dirs": "archive/2026-07-08_release_provenance_cleanup/stale_cmapss_smoke_figures/",
            "note": "TeX includes figures from manuscript/. Source/vector exports are intentionally retained under figures/.",
        },
        "figure_source_exports": [
            file_info("figures/fig_architecture.pdf"),
            file_info("figures/fig_risk.pdf"),
            file_info("figures/fig_calibration.pdf"),
            file_info("figures/fig_rul.pdf"),
            file_info("figures/fig_cmapss_external_metrics.pdf"),
            file_info("figures/fig_policy.pdf"),
            file_info("figures/fig_ablation.pdf"),
            file_info("figures/fig_attribution.pdf"),
        ],
        "key_file_hashes": {
            "manuscript/SCI-f08-predictive_maintenance_twin.tex": sha256(ROOT / "manuscript" / "SCI-f08-predictive_maintenance_twin.tex"),
            "manuscript/SCI-f08-predictive_maintenance_twin.pdf": sha256(ROOT / "manuscript" / "SCI-f08-predictive_maintenance_twin.pdf"),
            "manuscript/references.bib": sha256(ROOT / "manuscript" / "references.bib"),
            "results/policy_eval.json": sha256(ROOT / "results" / "policy_eval.json"),
            "results/q1_sensitivity.json": sha256(ROOT / "results" / "q1_sensitivity.json"),
            "results/q1_robustness.json": sha256(ROOT / "results" / "q1_robustness.json"),
            "results/q1_high_wear_sensitivity.json": sha256(ROOT / "results" / "q1_high_wear_sensitivity.json"),
            "results/q1_model_seed_audit.json": sha256(ROOT / "results" / "q1_model_seed_audit.json"),
            "results/stats_extra.json": sha256(ROOT / "results" / "stats_extra.json"),
            "results/cmapss_external_vps_20260614_121237/metric_summary.csv": sha256(ROOT / "results" / "cmapss_external_vps_20260614_121237" / "metric_summary.csv"),
        },
    }
    jdump(manifest, str(ROOT / "results" / "replication_manifest.json"))
    with (ROOT / "results" / "replication_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["artifact", "script", "command", "inputs", "outputs", "seeds", "notes"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for item in rows:
            w.writerow(item)
    exp_dir = ROOT / "experiments"
    exp_dir.mkdir(exist_ok=True)
    with (exp_dir / "EXPERIMENT_REGISTRY.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_id",
            "status",
            "artifact",
            "script",
            "command",
            "inputs",
            "outputs",
            "seeds",
            "provenance_source",
            "notes",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, item in enumerate(rows, start=1):
            w.writerow({
                "run_id": f"f08-registry-{i:02d}",
                "status": "verified_from_existing_artifacts",
                "artifact": item["artifact"],
                "script": item["script"],
                "command": item["command"],
                "inputs": item["inputs"],
                "outputs": item["outputs"],
                "seeds": item["seeds"],
                "provenance_source": "results/replication_manifest.json; MD/_state/PROVENANCE_AUDIT_20260707_RAW_TO_TABLE.md",
                "notes": item["notes"],
            })
    print("wrote replication manifest")


if __name__ == "__main__":
    main()

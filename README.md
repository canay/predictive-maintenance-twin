# Benchmark-Anchored Decision Environment with Mechanism-Aligned Gating for Predictive Maintenance

Reproducibility materials for the manuscript *A Benchmark-Anchored Decision
Environment with Mechanism-Aligned Gating for Predictive Maintenance*
(Özkan Canay, Sakarya University). **The manuscript is under review; it is not
included in this repository and no results here should be cited as published.**

This repository contains the analysis code, fixed seeds, run manifests, result
tables, prediction files, and figure/table provenance records behind every
number reported in the paper. It does **not** contain the manuscript sources,
the compiled article, or the third-party benchmark datasets.

## What the study evaluates

Four separable layers are evaluated against separate evidence sources rather
than as one end-to-end claim:

| Layer | Component | Evidence source |
|---|---|---|
| A1 | Failure-risk detector (LR / RF / HGB) | AI4I 2020 benchmark |
| A2 | Tool-life RUL prognostics (HGBR / MLP) | Run-to-failure simulator; C-MAPSS as a secondary temporal check |
| A3 | Cost-aware maintenance policy | Run-to-failure simulator |
| A4 | TreeSHAP mechanism gate on preventive actions | AI4I driver sets; simulator candidate cycles |

The maintenance-policy and gating results are simulator-bound. They do not
establish plant deployment readiness.

## Data (not redistributed)

Both benchmarks are public third-party datasets and must be obtained from their
original sources.

**AI4I 2020 Predictive Maintenance Dataset.** `code/stage0_data.py` downloads it
automatically from OpenML (`data_id=42890`) into `data/ai4i2020.csv` on first
run. Original sources: UCI Machine Learning Repository (DOI
`10.24432/C5HS5C`) and Matzka (2020), DOI `10.1109/AI4I49448.2020.00023`.

**NASA C-MAPSS turbofan degradation benchmark.** Download the four official
subsets from the NASA Prognostics Center of Excellence data repository and place
`train_FD00x.txt`, `test_FD00x.txt`, and `RUL_FD00x.txt` under `data/cmapss/`.
Reference: Saxena et al. (2008), DOI `10.1109/PHM.2008.4711414`.

## Environment

The recorded environment for the reported results (`results/versions.txt`):

```text
python 3.12.12   numpy 2.3.5    pandas 2.2.3    scikit-learn 1.7.2
scipy 1.17.1     matplotlib 3.10.8   shap 0.50.0   torch 2.10.0   joblib 1.5.3
```

Saved model artifacts were produced under scikit-learn 1.7.2. The crossed
model-seed audit was rerun in a dedicated 1.7.2 environment so that the stored
artifacts load without version warnings.

## Running the pipeline

Stages are resumable and write into `results/`. Run them in order from the
repository root:

```bash
python code/stage0_data.py           # fetch AI4I into data/
python code/stage1_risk.py           # A1 detector metrics (Table: risk)
python code/stage2_calib.py          # simulator calibration (Table: calib)
python code/stage3_rul.py            # tool-life RUL (Table: rul)
python code/stage4a_models.py        # freeze deployed risk/RUL models
python code/stage4b_gate.py          # TreeSHAP gate over candidate cycles
python code/stage4c_policy.py        # policy evaluation over 15 pools
python code/stage5_shap.py           # attribution fidelity and stability
python code/stage9_stats.py          # paired Wilcoxon tests
python code/stage10_q1_sensitivity.py
python code/stage12_q1_robustness.py # gate controls, strict-RUL, wear, oracle
python code/stage13_high_wear_sensitivity.py
python code/stage14_model_seed_audit.py
python code/stage11_replication_manifest.py
```

The C-MAPSS check is a separate run:

```bash
python code/run_cmapss_external_validation.py
```

`code/vps_run_cmapss_external_validation.sh` is the wrapper used for the
reported run on a 4-vCPU Linux host with two BLAS/joblib threads.

## Result-to-claim traceability

`results/replication_manifest.csv` and `results/replication_manifest.json` map
each reported table and figure to its generating script, command, input
artifacts, output artifacts, software environment, and seed set.

Key result files:

| File | Contents |
|---|---|
| `results/results_summary.json` | detector, RUL, policy and attribution summaries |
| `results/policy_eval.json` | per-pool cost rate, availability, failures, actions, false-action rates, Wilcoxon tests |
| `results/calibration.json` | simulator-versus-benchmark KS and Wasserstein distances, failure-mode rates |
| `results/shap_summary.json` | per-mode hit rates, Jaccard and Spearman stability |
| `results/gate_pe_*.npz` | raw per-candidate-cycle gate votes and decisions for the 15 evaluation pools |
| `results/q1_robustness.json` | stricter-RUL, raw-wear, wear-torque, fine-periodic, random and oracle controls |
| `results/q1_model_seed_audit.*` | crossed 5 model seeds by 15 pool seeds |
| `results/q1_high_wear_sensitivity.*` | high-wear warm-start stress test |
| `results/stats_extra.json` | savings-versus-reactive tests and bootstrap intervals |

`experiments/EXPERIMENT_REGISTRY.csv` lists every recorded run with its local
run folder, and each `experiments/<date>_<tool>_<location>_<slug>/` folder
carries its own manifest, status, CLI transcript, and evidence copies.

## Reproducing the headline numbers

The cost model is `C = N_p c_p + N_f c_c + c_d (N_p d_p + N_f d_c)` with
`c_p = 1`, `c_d = 0.05`, `d_p = 2`, `d_c = 10`, and each 400-episode pool has
exactly one terminal outcome per episode, so `N_p + N_f = 400`. Cost rate is
`C / op`, where `op` is the number of operating cycles simulated in the pool and
is policy dependent. The gate decision is a majority vote: a candidate cycle
passes when tool wear ranks in the top two absolute TreeSHAP values for at least
two of three gate models. Both quantities can be recomputed directly from the
per-seed arrays in `policy_eval.json` and the raw vote arrays in `gate_pe_*.npz`
without rerunning any model.

## Status

No archival DOI is minted for this repository unless the journal requests one.

A license has not been declared yet, so default copyright applies. Contact the
author before reuse.

## Contact

Özkan Canay, Department of Information Systems and Technologies, Faculty of
Computer and Information Sciences, Sakarya University, Sakarya, Türkiye
(canay@sakarya.edu.tr).

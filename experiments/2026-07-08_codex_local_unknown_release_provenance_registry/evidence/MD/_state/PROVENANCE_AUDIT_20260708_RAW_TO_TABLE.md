# Pre-submission Raw-to-table Provenance Audit

2026-07-08 15:45 +03:00;Antigravity;Gemini 3.5 Flash;20260708-f08-raw-to-table-provenance-audit

## Scope

Bu denetim deneyleri yeniden koşmadan, mevcut dosya/veri/kod/sonuç ve TeX yapısının tutarlılığını doğrulamak üzere yapılmıştır.

Stage move yapılmamıştır. `PROJECT_STATE.json` içindeki G10 durumu değiştirilmemiştir.

## Main Findings

✅ **verify_tex.py:** Çalıştırıldı ve sonuç:
```
OK prev339
OK prev_TWF
OK prev_HDF
OK prev_PWF
OK prev_OSF
OK prev_RNF
OK rho
OK sig
OK mu_a
OK twf_win
OK type_probs
OK mean_inc
OK rules_exact
OK riskrow_LR_Machine_failure
OK riskrow_RF_Machine_failure
OK riskrow_HGB_Machine_failure
OK riskrow_LR_TWF
OK riskrow_RF_TWF
OK riskrow_HGB_TWF
OK riskrow_LR_HDF
OK riskrow_RF_HDF
OK riskrow_HGB_HDF
OK riskrow_LR_PWF
OK riskrow_RF_PWF
OK riskrow_HGB_PWF
OK riskrow_LR_OSF
OK riskrow_RF_OSF
OK riskrow_HGB_OSF
OK riskrow_LR_RNF
OK riskrow_RF_RNF
OK riskrow_HGB_RNF
OK rf_mf
OK hgb_hdf
OK hgb_pwf
OK lr_osf
OK lr_twf
OK twf_ap_lt_015
OK n_pos_twf
OK rnf_chance
OK rf_vs_hgb_p
OK cal_pool
OK ks_tq
OK ks_rpm
OK ks_air
OK ks_proc
OK ks_wear
OK rate_HDF
OK rate_PWF
OK rate_RNF
OK rate_TWF
OK rate_OSF
OK wear_mu
OK calrow_air
OK calrow_proc
OK calrow_rpm
OK calrow_tq
OK calrow_wear
OK rulrow_HGBR
OK rulrow_MLP
OK rul_wilcoxon
OK deg_eplen
OK allmodes
OK allmodes_intex
OK polrow_r2_reactive
OK polrow_r2_periodic
OK polrow_r2_risk
OK polrow_r2_risk_gated
OK polrow_r2_rul
OK polrow_r2_rul_gated
OK polrow_r5_reactive
OK polrow_r5_periodic
OK polrow_r5_risk
OK polrow_r5_risk_gated
OK polrow_r5_rul
OK polrow_r5_rul_gated
OK polrow_r10_reactive
OK polrow_r10_periodic
OK polrow_r10_risk
OK polrow_r10_risk_gated
OK polrow_r10_rul
OK polrow_r10_rul_gated
OK polrow_r20_reactive
OK polrow_r20_periodic
OK polrow_r20_risk
OK polrow_r20_risk_gated
OK polrow_r20_rul
OK polrow_r20_rul_gated
OK pooled_risk_rul
OK pooled_reactive_rul
OK pooled_periodic
OK pooled_gate_cost
OK pooled_riskgate
OK gate_false30
OK sav_seq
OK sav_sd
OK risk_best_sav
OK prog_range
OK gate_cost_range
OK gate_fa
OK gate_pass
OK gate_pass_range
OK gate_cand_range
OK wear_avoid
OK avail_rul
OK f1000
OK abl_r2
OK abl_r5
OK abl_r10
OK abl_r20
OK shaprow_TWF
OK shaprow_HDF
OK shaprow_PWF
OK shaprow_OSF
OK shaprow_RNF
OK shap_hits
OK q1_title
OK q1_no_benchmark_calibrated
OK q1_partial_calibration
OK q1_cmapss_engine_ci
OK q1rob_row_2
OK q1rob_row_5
OK q1rob_row_10
OK q1rob_row_20
OK q1_candidate_gate
OK q1_false_thresholds
OK q1_high_wear_pool
OK q1_high_wear_costs
OK q1_high_wear_fa
OK q1_model_seed_costs
OK q1_model_seed_variation
OK q1_model_seed_gate

FAILURES: none
```

## Remaining G11 Risks

⚠️ **Kalan risk (P1):** Public GitHub repository, frozen release/commit, and Zenodo DOI are not yet real and must not be claimed.

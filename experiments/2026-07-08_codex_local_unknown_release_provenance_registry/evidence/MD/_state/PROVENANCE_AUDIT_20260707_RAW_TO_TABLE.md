# Pre-submission Raw-to-table Provenance Audit

2026-07-07 13:49 +03:00;Codex;GPT-5 Codex;20260707-f08-raw-to-table-provenance-audit

## Scope

Bu denetim deneyleri yeniden koşmadan yapıldı. Amaç, makalede raporlanan ana sayısal sonuçların, tablo değerlerinin ve figür iddialarının mevcut kod, result artefaktları, manifest ve TeX ile tutarlı olup olmadığını kontrol etmekti.

Stage move yapılmadı. `PROJECT_STATE.json` içinde G10 durumu değiştirilmedi.

## Files Read

- `C:\DOCS\AKADEMIK\AkademikAkisMerkezi\RUN.md`
- `C:\DOCS\AKADEMIK\AkademikAkisMerkezi\Akis1_AnaPipeline\START.md`
- `C:\DOCS\AKADEMIK\AkademikAkisMerkezi\Akis1_AnaPipeline\PIPELINE.md`
- `C:\DOCS\AKADEMIK\AkademikAkisMerkezi\Akis1_AnaPipeline\STATE_AND_HANDOFF.md`
- `C:\DOCS\AKADEMIK\AkademikAkisMerkezi\Akis1_AnaPipeline\STUDY_DESIGN_AND_EXPERIMENTS.md`
- `C:\DOCS\AKADEMIK\AkademikAkisMerkezi\Akis1_AnaPipeline\FIGURE_TABLE_VISUAL_QA.md`
- `C:\DOCS\AKADEMIK\AkademikAkisMerkezi\Akis1_AnaPipeline\REPRODUCIBILITY_AND_ARTIFACTS.md`
- `MD/_state/PROJECT_STATE.json`
- `MD/_state/HANDOFF.md`
- `MD/_state/QUALITY_LEDGER.md`
- `MD/_state/HUMAN_TASKS.md`
- `MD/_state/FILE_MANIFEST.md`
- `manuscript/SCI-f08-predictive_maintenance_twin.tex`
- `results/replication_manifest.json`
- `results/policy_eval.json`
- `results/results_summary.json`
- `results/stats_extra.json`
- `results/q1_sensitivity.json`
- `results/q1_robustness.json`
- `results/q1_model_seed_audit.json`
- `results/q1_high_wear_sensitivity.json`
- `results/cmapss_external_vps_20260614_121237/*`
- `code/stage4c_policy.py`
- `code/stage7_figs.py`
- `code/stage8_summary.py`
- `code/stage9b_stats.py`
- `code/stage10_q1_sensitivity.py`
- `code/stage11_replication_manifest.py`
- `code/stage12_q1_robustness.py`
- `code/stage14_model_seed_audit.py`
- `code/run_cmapss_external_validation.py`
- `code/plot_cmapss_external_validation.py`
- `code/verify_tex.py`

## Main Finding

✅ Tamamlandı: Mevcut TeX sayısal değerleri ile mevcut result JSON artefaktları arasında `code/verify_tex.py` düzeyinde uyuşmazlık bulunmadı. Komut çıktısı `FAILURES: none` verdi.

✅ Tamamlandı: Ana policy/gate değerlendirmesinin N=15 olduğu mevcut artefaktlarla doğrulandı. `results/policy_eval.json` içinde `eval_seeds` alanı 0--14 aralığında 15 evaluation pool içeriyor; `design_note` 15 independent simulator evaluation pool ve saved model seeds 0--4 cyclic reuse tasarımını açıkça kaydediyor.

✅ Tamamlandı: `results/q1_model_seed_audit.json` tasarımı, beş saved risk/RUL model seed'i ile 15 evaluation-pool seed'in çaprazlandığını gösteriyor. Bu, ana policy sonuçlarının sadece cyclic model/pool eşlemesine dayanmadığı yönündeki manuscript iddiasını destekliyor.

✅ Tamamlandı: C-MAPSS kanıtının scope'u TeX içinde prognostic/RUL layer ile sınırlı tutulmuş. Makale C-MAPSS'i maintenance policy veya TreeSHAP gate için plant-level validation gibi kullanmıyor; ilgili sınırlama Results, Discussion ve Limitations bölümlerinde açık.

## Source Mapping

| Manuscript object / claim family | Current source artefact(s) | Producing script(s) | Audit result |
|---|---|---|---|
| AI4I risk table and `fig_risk` | `results/risk/*.json`, `results/results_summary.json` | `code/stage0_data.py`, `code/stage1_risk.py`, `code/stage7_figs.py`, `code/gen_tables.py` | TeX values verified by `verify_tex.py`; five stratified seeds are expected for this layer. |
| Simulator calibration table and `fig_calibration` | `results/sim_params.json`, `results/calibration.json`, `results/results_summary.json` | `code/stage2_calib.py`, `code/stage7_figs.py`, `code/gen_tables.py` | TeX values verified by `verify_tex.py`; calibration pool is recorded as 600 episodes, seed 12345. |
| RUL prognostic table and `fig_rul` | `results/rul/*.json`, `results/rul_allmodes/*.json`, `results/stats_extra.json` | `code/stage3_rul.py`, `code/stage9b_stats.py`, `code/stage7_figs.py`, `code/gen_tables.py` | TeX values verified by `verify_tex.py`; five episode-level seeds are expected for RUL training/evaluation. |
| Policy table, policy figure, ablation table/figure | `results/policy_eval.json`, `results/stats_extra.json`, `results/gate_pe_*.npz` | `code/stage4a_models.py`, `code/stage4b_gate.py`, `code/stage4c_policy.py`, `code/stage9b_stats.py`, `code/stage7_figs.py`, `code/gen_tables.py` | N=15 evaluation pools verified; TeX policy rows verified by `verify_tex.py`. |
| Q1 sensitivity/gate-control tables | `results/q1_sensitivity.json/csv`, `results/q1_robustness.json/csv` | `code/stage10_q1_sensitivity.py`, `code/stage12_q1_robustness.py` | Current files exist and are dated 2026-07-03; TeX sensitivity strings checked by `verify_tex.py`. |
| Model-seed audit | `results/q1_model_seed_audit.json/csv` | `code/stage14_model_seed_audit.py` | 75-cell model-seed by pool-seed grid supported by JSON/CSV; TeX strings checked by `verify_tex.py`. |
| High-wear warm-start sensitivity | `results/q1_high_wear_sensitivity.json/csv` | `code/stage13_high_wear_sensitivity.py` | Auxiliary five-pool sensitivity, correctly described as auxiliary rather than main N=15 policy evidence. |
| Attribution fidelity table and `fig_attribution` | `results/shap/*.json`, `results/shap_summary.json` | `code/stage5_shap.py`, `code/stage5c_pos.py`, `code/stage5b_agg.py`, `code/stage7_figs.py` | TeX attribution rows verified by `verify_tex.py`; five AI4I split seeds are expected for this layer. |
| External C-MAPSS table/figure | `results/cmapss_external_vps_20260614_121237/*.csv/json/jsonl`, `logs/vps_cmapss_external_20260614_121237/*` | `code/run_cmapss_external_validation.py`, `code/plot_cmapss_external_validation.py` | Run manifest records FD001--FD004, models, seeds 0--2, 48 benchmark rows, resource report and logs. TeX C-MAPSS strings checked by `verify_tex.py`. |

## N=15 / N=5 Assessment

✅ Tamamlandı: Eski N=5 issue, ana simulator policy/gate evaluation için aktif görünmüyor. `policy_eval.json` içinde her ratio/policy için 15 per-seed cost-rate değeri var.

✅ Tamamlandı: TeX içinde "five seeds" kullanımı risk/RUL-training/attribution katmanları için kalıyor; bu katmanlarda beş seed tasarımı hâlâ doğru ve policy N=15 iddiasıyla karışmıyor.

⚠️ Kalan risk: High-wear warm-start stress test auxiliary five-pool sensitivity olarak kalıyor. Makale bunu ana policy evidence gibi sunmuyor; ancak final submission sırasında bu yardımcı testin iddia sınırı korunmalı.

## C-MAPSS Scope Assessment

✅ Tamamlandı: C-MAPSS, external temporal RUL/prognostic validation olarak kullanılıyor.

✅ Tamamlandı: TeX içinde C-MAPSS'in explanation gate veya maintenance policy için plant-data validation olmadığı açıkça belirtilmiş.

⚠️ Kalan risk: Bu sınırlama manuscript dilinde korunmalı; final polish sırasında C-MAPSS sonucunu "policy transfer", "deployment validation" veya "gate validation" gibi genişleten ifade eklenmemeli.

## Provenance Gaps

⚠️ Kalan risk: `results/replication_manifest.json` içindeki `key_file_hashes` bölümünde `manuscript/SCI-f08-predictive_maintenance_twin.tex` hash'i güncel TeX ile eşleşmiyor. Proje `MD/_state/FILE_MANIFEST.md` içindeki 2026-07-03 RESS seed-expanded TeX hash'i güncel dosya ile eşleşiyor; stale olan local replication manifest kopyasıdır. Final public package öncesi `code/stage11_replication_manifest.py` yeniden çalıştırılmalı veya manifest hash alanı güncellenmelidir.

⚠️ Kalan risk: Proje, mevcut Akış 1 standartlarının istediği `experiments/EXPERIMENT_REGISTRY.csv` ve `MD/05_experiments/experiment_registry_notes.md` katmanına tam sahip değil. Mevcut kanıt zinciri `results/replication_manifest.json`, result JSON/CSV/NPZ dosyaları, C-MAPSS run manifest/resource report/logs ve state ledger kayıtları üzerinden izlenebiliyor. Bu, mevcut TeX sayılarında doğrulama hatası üretmedi; fakat final GitHub/release paketinde pipeline-standard registry tablosuna dönüştürülmesi gerekir.

⚠️ Kalan risk: Bazı TeX-facing `manuscript/fig_*.png` dosyaları, `figures/` altındaki aynı adlı kaynak PNG'lerle birebir hash eşleşmiyor. `fig_policy.png` ve `fig_ablation.png` eşleşiyor; `fig_risk.png`, `fig_rul.png`, `fig_calibration.png`, `fig_attribution.png`, `fig_architecture.png` ve C-MAPSS figure dosyaları farklı export/copy sürümleri. Sayısal TeX doğrulaması geçtiği için bu bir numeric-claim blocker değil; final release package öncesi figure manifest ve TeX-facing figure source mapping yenilenmeli.

## Gate Interpretation

✅ Tamamlandı: Bu denetim aktif G10 numeric/provenance blocker bulmadı. G10 state'i değiştirilmedi.

⚠️ Kalan risk: G11 öncesi provenance residual risk tamamen kapanmış sayılmamalı. Public GitHub paketine çıkmadan önce replication manifest hash'leri, figure source mapping ve experiment-registry tarzı run provenance tablosu temizlenmeli.

⚠️ Sizin kararınız gerekiyor: Bu çalışma artık basit taslak görünmüyor; ancak yazarın talimatı gereği `20_TASLAK_CALISMALAR` altında bırakıldı ve `40_AKTIF_CALISMALAR` aktarımı yapılmadı.

## Recommendation

Bu audit sonucuna göre deneyleri yeniden koşmak zorunlu görünmüyor. En uygun sonraki güvenli adım, G11 öncesi "release provenance cleanup" olmalı:

1. `code/stage11_replication_manifest.py` yeniden çalıştırılıp `results/replication_manifest.json/csv` güncel TeX hash'iyle tazelenmeli.
2. `experiments/EXPERIMENT_REGISTRY.csv` veya eşdeğer `MD/_state` provenance table, mevcut result manifestinden türetilmeli.
3. TeX-facing figure dosyaları için `figures/` source ile `manuscript/fig_*` output mapping açıkça kaydedilmeli.
4. Sonrasında `code/verify_tex.py`, Akış 1 validators ve `audit.py` tekrar çalıştırılmalı.

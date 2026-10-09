# Supplementary Data S8: Analysis Code and Manifest

Repository: `https://github.com/zhanfbiting/Evaluation-of-LLM-for-bladder-cancer-education`

## Canonical analysis code

- `analysis_code/analysis_pipeline.py`: mDISCERN, readability, risk-theme, AHP, WRS, and exploratory analyses.
- `analysis_code/guideline_concordance_analysis.py`: clinical-content subtotal, evidence source/timeliness, and paired Bonferroni comparisons.
- `analysis_code/sensitivity_analysis.py`: alternative AHP weights and percentile thresholds.

## Derived files

Readability is reproduced by running `python rebuild_readability.py` from `analysis_code/`. Original response files are in `source_data/model_outputs/*_raw.docx`; the supplied shared glossary is `source_data/model_outputs/medical_glossary.txt`. Historical segmented filenames are explicitly mapped in `config.py` and used for identity audit only. Character coverage and R are calculated on complete original answers; no product-specific calibration is used.

- `Supplementary_Data_S8_readability_response_level.csv`
- `Supplementary_Data_S8_readability_components_summary.csv`
- `Supplementary_Data_S8_readability_friedman.csv`
- `Supplementary_Data_S8_readability_pairwise.csv`
- `Supplementary_Data_S8_readability_quality_correlation.csv`
- `Supplementary_Data_S8_readability_input_audit.csv`

- `Supplementary_Data_S2_guideline_response_level_recalculated.csv`
- `Supplementary_Data_S2_guideline_model_summary_recalculated.csv`
- `Supplementary_Data_S2_guideline_pairwise_recalculated.csv`
- `Supplementary_Data_S5_mDISCERN_scoring_outputs.csv`
- `Supplementary_Data_S6_risk_theme_summary.csv`
- `Supplementary_Data_S6_risk_theme_cochran_q.csv`
- `Supplementary_Data_S6_risk_theme_pairwise_exact_mcnemar.csv`
- `Supplementary_Data_S6_AHP_weights.csv`
- `Supplementary_Data_S6_AHP_consistency.csv`
- `Supplementary_Data_S7_sensitivity_scenarios.csv`
- `Supplementary_Data_S7_primary_case_retention.csv`
- `Supplementary_Data_S7_ranking_correlations.csv`

## Collection and source-timing notes

The hot-search question set was collected as a one-time snapshot on November 2, 2025; a weekly export frequency is therefore not applicable. Guideline-related scoring used the documented source versions available before the March 10, 2026 response-collection cutoff, as listed in Supplementary Data S2.

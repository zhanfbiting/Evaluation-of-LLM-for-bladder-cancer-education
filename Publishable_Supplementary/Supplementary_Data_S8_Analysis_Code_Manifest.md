# Supplementary Data S8: Analysis Code and Manifest

Repository: `https://github.com/zhanfbiting/Evaluation-of-LLM-for-bladder-cancer-education`

## Canonical analysis code

- `analysis_code/analysis_pipeline.py`: mDISCERN, readability, risk-theme, AHP, WRS, and exploratory analyses.
- `analysis_code/guideline_concordance_analysis.py`: clinical-content subtotal, evidence source/timeliness, and paired Bonferroni comparisons.
- `analysis_code/sensitivity_analysis.py`: alternative AHP weights and percentile thresholds.

## Derived files

Readability is reproduced by running `python rebuild_readability.py` from `analysis_code/`. Original response files are in `source_data/model_outputs/*_raw.docx`; the supplied shared glossary is `source_data/model_outputs/medical_glossary.txt`. Character coverage and R are calculated directly from the complete original answers; no product-specific calibration is used. All submitted response-level outputs use the paper-question IDs.

- `Supplementary_Data_S8_readability_response_level.csv`
- `Supplementary_Data_S8_readability_components_summary.csv`
- `Supplementary_Data_S8_readability_friedman.csv`
- `Supplementary_Data_S8_readability_pairwise.csv`
- `Supplementary_Data_S8_readability_quality_correlation.csv`

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

The platform data were retrieved in one export on November 2, 2025. The records cover November 2024 through October 2025 and are organized by week; the study team did not make separate weekly exports. Repeated phrases across different weeks were retained in raw occurrence counts, while exact or semantically equivalent expressions were merged separately for unique-entry counts. The 1,713 related-search lines, 41 exclusions, and 1,672 screening entries are reported in Supplementary Table S1.

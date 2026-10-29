# Supplementary File Index

S1–S8 identify the supplementary groups used in the manuscript. Each group may contain more than one existing file. Paths below are relative to this Supplementary_Materials directory; worksheet names are exact.

| Group | Actual file / worksheet | Contents and manuscript use |
|---|---|---|
| S1 | `Supplementary_Table_S1_hot_search_to_question_mapping.csv` | Original search terms, mappings, occurrence and unique-entry counts; question-set construction and Table 1 |
| S1 | `Supplementary_Table_S1_metadata.md` | Retrieval date (2025-11-02), platform-record coverage (November 2024--October 2025), weekly record structure, duplicate-counting rules, and screening counts |
| S2 | `Supplementary_Data_S2_Guideline_Concordance.xlsx`: `S7_Reference_standards`, `S9_Rubric`, `S10_Question_specific_criteria` | Question-specific expected content, source references and 0/1/2 criteria; guideline-related Methods |
| S2 | Same workbook: `S1_Questions`, `S2_LLM_responses`, `S3_GC_scores` | Questions, 60 Chinese responses, source-question crosswalk, and both raters' item and total scores |
| S2 | Same workbook: `S4_Model_summary`, `S5_Question_summary`, `S6_Reliability`, `S8_Sources` | Product/question summaries, reliability and source inventory; Tables 9/10 |
| S2 | `Publishable_Supplementary/Supplementary_Data_S2_guideline_response_level_recalculated.csv` | Four-domain clinical subtotal, separate evidence/timeliness and full composite per response |
| S2 | `Publishable_Supplementary/Supplementary_Data_S2_guideline_model_summary_recalculated.csv` | Product-level mean/SD crosswalk; Table 9 |
| S2 | `Publishable_Supplementary/Supplementary_Data_S2_guideline_pairwise_recalculated.csv` | Paired tests and Bonferroni comparisons |
| S3 | `Supplementary_Table_S3_Nursing_Sensitive_Question_Analysis.xlsx`: `S1_Classification`, `S2_Group_summary`, `S3_Question_metrics`, `S4_Notes` | Five reporting categories, nursing-sensitive designation, exploratory summaries and notes |
| S4 | `Supplementary_Table_S4_Question_Set_Content_Validity_Ratings.xlsx`: `说明`, `专家评分`, `结果汇总` | CVI assessment instructions, individual ratings and question-domain summary values |
| S4 | `Publishable_Supplementary/Supplementary_Data_S4_CVI_question_level.csv`, `Publishable_Supplementary/Supplementary_Data_S4_CVI_summary.csv` | English question-level I-CVI values and domain/overall S-CVI/Ave verification tables |
| S5 | `ai_generated_content_quality_scoring_form.xlsx`: `Sheet_20260119` | mDISCERN items and score ranges |
| S5 | `Publishable_Supplementary/Supplementary_Data_S5_mDISCERN_scoring_outputs.csv` | Final q1–q8 and total scores, strengths and content concerns for all 60 responses; `question_id` uses paper numbering |
| S5 | `Publishable_Supplementary/Supplementary_Data_S5_mDISCERN_model_summary.csv`, `Publishable_Supplementary/Supplementary_Data_S5_mDISCERN_item_summary.csv` | English product-level and item-level verification summaries |
| S5 | `reviewer_scoring_outputs_clean.jsonl` | Structured response-level scoring input using the paper-question numbering |
| S6 | `risks_mapping.csv`, `strengths_mapping.csv` | Mapping of recorded observations to standardized descriptions |
| S6 | `AHP_expert_judgment.xlsx`: `Risk_themes`, `Saaty_scale`, `expert_1`, `expert_2`, `expert_3`, `Current_group_geometric_mean`, `Weights_and_CR`, `Final_group_weights` | Theme definitions, judgment matrices, group weights and consistency metrics; Table 2 |
| S6 | `Intervention Plan Expert Consultation Questionnaire - Kuang Jianying.docx`, `Intervention Plan Expert Consultation Questionnaire - Liu Yi.docx`, `Intervention Plan Expert Consultation Questionnaire - Tian Jun.docx` | Expert qualification and consultation questionnaires supporting the AHP panel description |
| S6 | `Publishable_Supplementary/Supplementary_Data_S6_risk_theme_summary.csv` | Theme occurrence frequencies; Table 6 |
| S6 | `Publishable_Supplementary/Supplementary_Data_S6_risk_theme_cochran_q.csv` | Cochran's Q statistic and P value for each binary risk theme |
| S6 | `Publishable_Supplementary/Supplementary_Data_S6_risk_theme_pairwise_exact_mcnemar.csv` | All three exact McNemar comparisons for each risk theme, with Bonferroni-adjusted P values; risk-theme Results |
| S6 | `Publishable_Supplementary/Supplementary_Data_S6_AHP_weights.csv` | AHP weights and ranking |
| S6 | `Publishable_Supplementary/Supplementary_Data_S6_AHP_consistency.csv` | Expert and group consistency indices |
| S7 | `analysis_code/outputs/tables/risk_threshold_summary.csv`, `analysis_code/outputs/tables/risk_quadrant_summary.csv`, `analysis_code/outputs/tables/risk_quadrant_by_model.csv`, `analysis_code/outputs/tables/high_score_high_risk_cases.csv` | WRS thresholds, quadrants and case identification; Table 7 |
| S7 | `Publishable_Supplementary/Supplementary_Data_S7_sensitivity_scenarios.csv` | Alternative weight/percentile scenarios; Table 8 |
| S7 | `Publishable_Supplementary/Supplementary_Data_S7_primary_case_retention.csv` | Main-case retention across scenarios |
| S7 | `Publishable_Supplementary/Supplementary_Data_S7_ranking_correlations.csv` | Rank correlations across weighting schemes |
| S8 | `analysis_code/README.md`, `analysis_code/requirements.txt`, `analysis_code/run_analysis.py`, `analysis_code/rebuild_readability.py`, `analysis_code/guideline_concordance_analysis.py`, `analysis_code/sensitivity_analysis.py` | Code entry points, dependencies and execution instructions |
| S8 | `Publishable_Supplementary/Supplementary_Data_S8_Analysis_Code_Manifest.md` | Analysis and derived-file manifest |
| S8 | `source_data/model_outputs/deepseek_raw.docx`, `source_data/model_outputs/doubao_raw.docx`, `source_data/model_outputs/yuanbao_raw.docx`, `source_data/model_outputs/medical_glossary.txt` | Original readability input text and common glossary |

## Readability outputs in S8

| Actual file | Contents and manuscript use |
|---|---|
| `Publishable_Supplementary/Supplementary_Data_S8_readability_response_level.csv` | Complete answer texts and C, sentence count, S, medical-term characters, T and R; `question_id` uses paper numbering; Tables 3/5 and Figure 4 |
| `Publishable_Supplementary/Supplementary_Data_S8_readability_components_summary.csv` | Product means and sample SDs; Tables 3/5 |
| `Publishable_Supplementary/Supplementary_Data_S8_readability_friedman.csv` | Paired omnibus test |
| `Publishable_Supplementary/Supplementary_Data_S8_readability_pairwise.csv` | All three pairwise tests and Bonferroni P values |
| `Publishable_Supplementary/Supplementary_Data_S8_readability_quality_correlation.csv` | Overall and product-specific Spearman correlations |

The reference to an S5, S6, S7 or S8 group does not imply that a combined workbook or ZIP with that group name exists. The exact files above provide the locations for the data and scoring records.

All submitted response-level tables use the paper-question numbering used in the manuscript.

The expert panel's professional roles and experience are described in the manuscript. The three expert qualification and consultation questionnaires listed under S6 are included in the public supplementary package; the numbered AHP judgment matrices and derived results are provided in `AHP_expert_judgment.xlsx` and the S6 verification files.

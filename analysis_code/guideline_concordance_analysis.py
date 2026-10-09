"""Recalculate guideline-concordance summaries from Supplementary Data S2.

The clinical-content subtotal is the sum of the four clinical domains. The
evidence-source/timeliness domain is exported separately because it represents
evidence traceability rather than clinical concordance. Pairwise tests are
paired by question and use Bonferroni correction across the three products.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats


CODE_DIR = Path(__file__).resolve().parent
WORKBOOK_CANDIDATES = [
    CODE_DIR.parent / "Supplementary_Data_S2_Guideline_Concordance.xlsx",
    CODE_DIR.parent / "Supplementary_Materials" / "Supplementary_Data_S2_Guideline_Concordance.xlsx",
]
WORKBOOK = next((path for path in WORKBOOK_CANDIDATES if path.exists()), WORKBOOK_CANDIDATES[-1])
OUTPUT_DIR = CODE_DIR / "outputs" / "tables"
MODEL_ORDER = ["DeepSeek", "Doubao", "Yuanbao"]
CLINICAL_COLUMNS = [
    "Mean accuracy",
    "Mean completeness",
    "Mean safety boundary",
    "Mean individualization",
]


def _bonferroni_pairwise(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    rows = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        paired = wide[[model_a, model_b]].dropna()
        differences = paired[model_a].to_numpy() - paired[model_b].to_numpy()
        if (differences == 0).all():
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(
                paired[model_a], paired[model_b], alternative="two-sided", method="auto"
            )
        rows.append(
            {
                "outcome": value_name,
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": len(paired),
                "statistic": statistic,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_bonferroni"] = (result["p_value"] * len(result)).clip(upper=1.0)
    return result


def main() -> None:
    if not WORKBOOK.exists():
        raise FileNotFoundError(WORKBOOK)

    scores = pd.read_excel(WORKBOOK, sheet_name="S3_GC_scores")
    scores["clinical_content_subtotal"] = scores[CLINICAL_COLUMNS].sum(axis=1)
    scores["full_composite_0_10"] = scores["Mean total"]
    scores["evidence_source_timeliness"] = scores["Mean evidence/timeliness"]

    model_summary = (
        scores.groupby("Model", sort=False)
        .agg(
            n=("Question ID", "size"),
            clinical_content_mean=("clinical_content_subtotal", "mean"),
            clinical_content_sd=("clinical_content_subtotal", "std"),
            full_composite_mean=("full_composite_0_10", "mean"),
            full_composite_sd=("full_composite_0_10", "std"),
            evidence_source_timeliness_mean=("evidence_source_timeliness", "mean"),
            evidence_source_timeliness_sd=("evidence_source_timeliness", "std"),
        )
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    wide_clinical = scores.pivot(index="Question ID", columns="Model", values="clinical_content_subtotal").reindex(columns=MODEL_ORDER)
    wide_evidence = scores.pivot(index="Question ID", columns="Model", values="evidence_source_timeliness").reindex(columns=MODEL_ORDER)
    pairwise = pd.concat(
        [
            _bonferroni_pairwise(wide_clinical, "clinical_content_subtotal_0_8"),
            _bonferroni_pairwise(wide_evidence, "evidence_source_timeliness_0_2"),
        ],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_DIR / "guideline_concordance_response_level_recalculated.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUTPUT_DIR / "guideline_concordance_model_summary_recalculated.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUTPUT_DIR / "guideline_concordance_pairwise_recalculated.csv", index=False, encoding="utf-8-sig")
    print(f"Saved guideline-concordance recalculations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

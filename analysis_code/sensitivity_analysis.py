from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
from analysis_pipeline import _calculate_ahp_metrics, _validate_ahp_matrix
from data_processing import build_label_table, build_question_meta, load_score_table


OUTPUT_DIR = config.CODE_DIR / "sensitivity_analysis"
THRESHOLDS = [0.70, 0.75, 0.80]
PRIMARY_WEIGHT_SCHEME = "AHP group"
PRIMARY_THRESHOLD = 0.75


def _risk_incidence_table(label_df: pd.DataFrame) -> pd.DataFrame:
    risk_df = (
        label_df[label_df["label_type"] == "risk"]
        .drop_duplicates(subset=["model_under_evaluation", "question_id", "label"])
        .copy()
    )
    risk_df["flag"] = 1
    incidence = risk_df.pivot_table(
        index=["model_under_evaluation", "model_label", "question_id", "question_text"],
        columns="label",
        values="flag",
        fill_value=0,
        aggfunc="max",
    )
    incidence = incidence.reindex(columns=config.RISK_THEME_ORDER, fill_value=0).reset_index()
    return incidence


def _weight_schemes() -> pd.DataFrame:
    labels = list(config.RISK_THEME_ORDER)
    n = len(labels)
    rows = []
    expert_matrices = {
        expert_name: np.array(matrix_values, dtype=float)
        for expert_name, matrix_values in config.AHP_EXPERT_PAIRWISE_MATRICES.items()
    }
    matrix_stack = []
    for expert_name, matrix in expert_matrices.items():
        _validate_ahp_matrix(expert_name, matrix, n)
        matrix_stack.append(matrix)
        weights, *_ = _calculate_ahp_metrics(matrix)
        for label, weight in zip(labels, weights):
            rows.append({"weight_scheme": expert_name, "label": label, "weight": weight})

    group_matrix = np.exp(np.mean(np.log(np.stack(matrix_stack)), axis=0))
    _validate_ahp_matrix("AHP group", group_matrix, n)
    group_weights, *_ = _calculate_ahp_metrics(group_matrix)
    for label, weight in zip(labels, group_weights):
        rows.append({"weight_scheme": PRIMARY_WEIGHT_SCHEME, "label": label, "weight": weight})

    equal_weight = 1.0 / n
    for label in labels:
        rows.append({"weight_scheme": "Equal weights", "label": label, "weight": equal_weight})

    weight_df = pd.DataFrame(rows)
    weight_df["rank"] = weight_df.groupby("weight_scheme")["weight"].rank(method="first", ascending=False).astype(int)
    weight_df = weight_df.sort_values(["weight_scheme", "rank"]).reset_index(drop=True)
    return weight_df


def _score_responses(
    score_df: pd.DataFrame,
    question_meta_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
    weight_df: pd.DataFrame,
) -> pd.DataFrame:
    base_df = score_df[
        ["model_under_evaluation", "model_label", "question_id", "question_text", "total"]
    ].merge(
        question_meta_df[["question_id", "paper_question_id", "question_short"]],
        on="question_id",
        how="left",
    )
    risk_flags = incidence_df[
        ["model_under_evaluation", "model_label", "question_id", "question_text"] + list(config.RISK_THEME_ORDER)
    ]
    base_df = base_df.merge(
        risk_flags,
        on=["model_under_evaluation", "model_label", "question_id", "question_text"],
        how="left",
    )
    base_df[config.RISK_THEME_ORDER] = base_df[config.RISK_THEME_ORDER].fillna(0).astype(int)

    scored_frames = []
    for weight_scheme, subset in weight_df.groupby("weight_scheme", sort=False):
        weights = subset.set_index("label").reindex(config.RISK_THEME_ORDER)["weight"].to_numpy(dtype=float)
        scored = base_df.copy()
        scored["weight_scheme"] = weight_scheme
        scored["weighted_risk_score"] = scored[config.RISK_THEME_ORDER].to_numpy(dtype=float).dot(weights)
        scored["risk_label_count"] = scored[config.RISK_THEME_ORDER].sum(axis=1)
        scored_frames.append(scored)
    return pd.concat(scored_frames, ignore_index=True)


def _classify_scenarios(scored_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_cutoff = float(scored_df[scored_df["weight_scheme"] == PRIMARY_WEIGHT_SCHEME]["total"].median())
    scenario_rows = []
    case_frames = []

    for weight_scheme, scheme_df in scored_df.groupby("weight_scheme", sort=False):
        for percentile in THRESHOLDS:
            risk_cutoff = float(scheme_df["weighted_risk_score"].quantile(percentile))
            classified = scheme_df.copy()
            classified["threshold_percentile"] = int(percentile * 100)
            classified["quality_cutoff"] = quality_cutoff
            classified["risk_cutoff"] = risk_cutoff
            classified["high_score"] = classified["total"] >= quality_cutoff
            classified["high_risk"] = classified["weighted_risk_score"] >= risk_cutoff
            classified["high_score_high_risk"] = classified["high_score"] & classified["high_risk"]
            classified["case_id"] = (
                classified["model_label"]
                + "-Q"
                + classified["paper_question_id"].astype(int).astype(str)
            )

            high_risk_n = int(classified["high_risk"].sum())
            hshr_df = classified[classified["high_score_high_risk"]].copy()
            case_list = "; ".join(hshr_df.sort_values(["paper_question_id", "model_label"])["case_id"].tolist())
            scenario_rows.append(
                {
                    "weight_scheme": weight_scheme,
                    "threshold_percentile": int(percentile * 100),
                    "quality_cutoff": quality_cutoff,
                    "risk_cutoff": risk_cutoff,
                    "high_risk_n": high_risk_n,
                    "high_score_high_risk_n": len(hshr_df),
                    "high_score_high_risk_cases": case_list,
                }
            )
            case_frames.append(
                hshr_df[
                    [
                        "weight_scheme",
                        "threshold_percentile",
                        "case_id",
                        "model_under_evaluation",
                        "model_label",
                        "question_id",
                        "paper_question_id",
                        "question_short",
                        "total",
                        "weighted_risk_score",
                        "risk_label_count",
                    ]
                ]
            )

    scenario_df = pd.DataFrame(scenario_rows).sort_values(
        ["threshold_percentile", "weight_scheme"]
    ).reset_index(drop=True)
    case_df = pd.concat(case_frames, ignore_index=True).sort_values(
        ["threshold_percentile", "weight_scheme", "paper_question_id", "model_label"]
    ).reset_index(drop=True)
    return scenario_df, case_df


def _primary_case_retention(scenario_case_df: pd.DataFrame) -> pd.DataFrame:
    primary_cases = set(
        scenario_case_df[
            (scenario_case_df["weight_scheme"] == PRIMARY_WEIGHT_SCHEME)
            & (scenario_case_df["threshold_percentile"] == int(PRIMARY_THRESHOLD * 100))
        ]["case_id"]
    )
    rows = []
    for (weight_scheme, threshold), subset in scenario_case_df.groupby(
        ["weight_scheme", "threshold_percentile"], sort=False
    ):
        cases = set(subset["case_id"])
        retained = sorted(primary_cases & cases)
        rows.append(
            {
                "weight_scheme": weight_scheme,
                "threshold_percentile": threshold,
                "primary_case_n": len(primary_cases),
                "retained_primary_case_n": len(retained),
                "retained_primary_case_rate": len(retained) / len(primary_cases) if primary_cases else np.nan,
                "retained_primary_cases": "; ".join(retained),
                "new_cases": "; ".join(sorted(cases - primary_cases)),
            }
        )
    return pd.DataFrame(rows).sort_values(["threshold_percentile", "weight_scheme"]).reset_index(drop=True)


def _ranking_correlations(scored_df: pd.DataFrame) -> pd.DataFrame:
    pivot = scored_df.pivot_table(
        index=["model_under_evaluation", "question_id"],
        columns="weight_scheme",
        values="weighted_risk_score",
    )
    primary = pivot[PRIMARY_WEIGHT_SCHEME]
    rows = []
    for weight_scheme in pivot.columns:
        rho, p_value = stats.spearmanr(primary, pivot[weight_scheme])
        rows.append(
            {
                "comparison": f"{PRIMARY_WEIGHT_SCHEME} vs {weight_scheme}",
                "weight_scheme": weight_scheme,
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("weight_scheme").reset_index(drop=True)


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    score_df = load_score_table()
    question_meta_df = build_question_meta()
    label_df = build_label_table(score_df)
    incidence_df = _risk_incidence_table(label_df)
    weight_df = _weight_schemes()
    scored_df = _score_responses(score_df, question_meta_df, incidence_df, weight_df)
    scenario_df, scenario_case_df = _classify_scenarios(scored_df)
    retention_df = _primary_case_retention(scenario_case_df)
    correlation_df = _ranking_correlations(scored_df)

    outputs = {
        "scenario_summary": scenario_df,
        "primary_case_retention": retention_df,
        "ranking_correlations": correlation_df,
    }
    for name, df in outputs.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    print(f"Saved sensitivity-analysis tables to {OUTPUT_DIR}")
    for name in outputs:
        print(f"- {name}.csv")


if __name__ == "__main__":
    main()

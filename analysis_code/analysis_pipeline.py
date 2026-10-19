import numpy as np
import pandas as pd

import config
from statistics_utils import (
    cochran_q_test,
    friedman_test,
    pairwise_exact_mcnemar,
    pairwise_wilcoxon,
    spearman_table,
    summarize_by_model,
)


def overall_quality_analysis(score_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = summarize_by_model(score_df, "total")
    wide_df, friedman_df = friedman_test(score_df, index_col="question_id", value_col="total")
    pairwise_df = pairwise_wilcoxon(wide_df)
    return {
        "overall_quality_summary": summary,
        "overall_quality_friedman": friedman_df,
        "overall_quality_pairwise_wilcoxon": pairwise_df,
    }


def mdiscern_item_analysis(score_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    long_df = score_df.melt(
        id_vars=["model_under_evaluation", "model_label", "question_id", "question_text"],
        value_vars=list(config.QUESTION_ITEM_LABELS.keys()),
        var_name="item_code",
        value_name="score",
    )
    long_df["item_label"] = long_df["item_code"].map(config.QUESTION_ITEM_LABELS)

    summary = (
        long_df.groupby(["item_code", "item_label", "model_under_evaluation", "model_label"], observed=True)
        .agg(
            均值=("score", "mean"),
            标准差=("score", "std"),
            中位数=("score", "median"),
        )
        .reset_index()
    )
    summary["model_under_evaluation"] = pd.Categorical(
        summary["model_under_evaluation"],
        categories=config.MODEL_ID_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["item_code", "model_under_evaluation"]).reset_index(drop=True)

    friedman_rows = []
    pairwise_frames = []
    for item_code, item_label in config.QUESTION_ITEM_LABELS.items():
        item_df = long_df[long_df["item_code"] == item_code]
        wide_df, friedman_df = friedman_test(item_df, index_col="question_id", value_col="score")
        friedman_df.insert(0, "item_code", item_code)
        friedman_df.insert(1, "item_label", item_label)
        friedman_rows.append(friedman_df)

        pairwise_df = pairwise_wilcoxon(wide_df)
        pairwise_df.insert(0, "item_code", item_code)
        pairwise_df.insert(1, "item_label", item_label)
        pairwise_frames.append(pairwise_df)

    return {
        "mdiscern_item_summary": summary,
        "mdiscern_item_friedman": pd.concat(friedman_rows, ignore_index=True),
        "mdiscern_item_pairwise_wilcoxon": pd.concat(pairwise_frames, ignore_index=True),
    }


def question_difficulty_analysis(score_df: pd.DataFrame, question_meta_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = (
        score_df.groupby("question_id", as_index=False)
        .agg(
            question_text=("question_text", "first"),
            平均总分=("total", "mean"),
            中位总分=("total", "median"),
            最低分=("total", "min"),
            最高分=("total", "max"),
            标准差=("total", "std"),
        )
        .assign(分歧极差=lambda df: df["最高分"] - df["最低分"])
        .merge(question_meta_df, on=["question_id", "question_text"], how="left")
        .sort_values(["平均总分", "分歧极差"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary["平均总分排名"] = summary["平均总分"].rank(method="dense", ascending=True).astype(int)
    summary["分歧极差排名"] = summary["分歧极差"].rank(method="dense", ascending=False).astype(int)

    average_cutoff = summary["平均总分"].median()
    disagreement_cutoff = summary["分歧极差"].median()
    summary["难度分层"] = summary["平均总分"].apply(lambda value: "低分" if value < average_cutoff else "高分")
    summary["分歧分层"] = summary["分歧极差"].apply(lambda value: "高分歧" if value >= disagreement_cutoff else "低分歧")
    summary["问题象限"] = summary["难度分层"] + "-" + summary["分歧分层"]

    heatmap_df = (
        score_df.merge(question_meta_df[["question_id", "paper_question_id"]], on="question_id", how="left")
        .pivot(index="paper_question_id", columns="model_label", values="total")
        .sort_index()
        .reindex(columns=[config.MODEL_LABEL_MAP[item] for item in config.MODEL_ID_ORDER])
    )
    heatmap_df.index = heatmap_df.index.map(lambda value: str(int(value)))
    quadrant_rows = []
    for quadrant_name, subset in summary.groupby("问题象限", sort=False):
        ordered_subset = subset.sort_values("paper_question_id")
        quadrant_rows.append(
            {
                "问题象限": quadrant_name,
                "题目数": len(ordered_subset),
                "题号": "、".join(str(int(value)) for value in ordered_subset["paper_question_id"]),
                "简短题名": "；".join(ordered_subset["question_short"]),
            }
        )
    quadrant_summary = pd.DataFrame(quadrant_rows).sort_values("题目数", ascending=False).reset_index(drop=True)
    return {
        "question_difficulty_summary": summary,
        "question_difficulty_quadrant_summary": quadrant_summary,
        "question_score_heatmap": heatmap_df,
    }


def risk_theme_analysis(label_df: pd.DataFrame, question_meta_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    risk_df = label_df[label_df["label_type"] == "risk"].copy()
    merged = risk_df.merge(question_meta_df, on=["question_id", "question_text"], how="left")
    incidence_base = merged.drop_duplicates(subset=["question_id", "model_under_evaluation", "label"]).copy()

    labels = list(config.RISK_THEME_ORDER)
    question_ids = question_meta_df["question_id"].drop_duplicates().tolist()
    full_index = pd.MultiIndex.from_product(
        [question_ids, labels],
        names=["question_id", "label"],
    )

    summary = (
        incidence_base.groupby(["label", "model_under_evaluation", "model_label"], as_index=False)
        .agg(出现频数=("question_id", "count"))
        .set_index(["label", "model_under_evaluation"])
        .reindex(
            pd.MultiIndex.from_product(
                [labels, config.MODEL_ID_ORDER],
                names=["label", "model_under_evaluation"],
            ),
            fill_value=0,
        )
        .reset_index()
    )
    summary["model_label"] = summary["model_under_evaluation"].map(config.MODEL_LABEL_MAP)
    summary["出现频数"] = summary["出现频数"].astype(int)
    summary["占比"] = summary["出现频数"] / len(config.QUESTION_META)
    overall_order = (
        summary.groupby("label", as_index=False)["出现频数"]
        .sum()
        .sort_values(["出现频数", "label"], ascending=[False, True])["label"]
        .tolist()
    )
    summary["label"] = pd.Categorical(summary["label"], categories=overall_order, ordered=True)
    summary["model_under_evaluation"] = pd.Categorical(
        summary["model_under_evaluation"],
        categories=config.MODEL_ID_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["label", "model_under_evaluation"]).reset_index(drop=True)

    incidence = (
        incidence_base.assign(flag=1)
        .pivot_table(
            index=["question_id", "label"],
            columns="model_under_evaluation",
            values="flag",
            fill_value=0,
        )
        .reindex(index=full_index, fill_value=0)
        .reindex(columns=config.MODEL_ID_ORDER)
        .reset_index()
    )

    cochran_rows = []
    pairwise_frames = []
    for label_name in labels:
        subset = incidence[incidence["label"] == label_name]
        binary_df = subset[config.MODEL_ID_ORDER]
        statistic, p_value = cochran_q_test(binary_df)
        cochran_rows.append(
            {
                "label": label_name,
                "n_questions": len(subset),
                "statistic": statistic,
                "p_value": p_value,
            }
        )
        pairwise_df = pairwise_exact_mcnemar(binary_df)
        pairwise_df.insert(0, "label", label_name)
        pairwise_frames.append(pairwise_df)

    return {
        "risk_theme_summary": summary,
        "risk_theme_cochran_q": pd.DataFrame(cochran_rows).sort_values("p_value", na_position="last"),
        "risk_theme_pairwise_exact_mcnemar": pd.concat(pairwise_frames, ignore_index=True),
    }


def _calculate_ahp_metrics(matrix: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[max_index].real)
    weights = np.abs(eigvecs[:, max_index].real)
    weights = weights / weights.sum()
    n = matrix.shape[0]
    consistency_index = (lambda_max - n) / (n - 1)
    random_index_map = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }
    random_index = random_index_map[n]
    consistency_ratio = consistency_index / random_index if random_index else 0.0
    return weights, lambda_max, consistency_index, random_index, consistency_ratio


def _validate_ahp_matrix(expert_name: str, matrix: np.ndarray, n: int) -> None:
    if matrix.shape != (n, n):
        raise ValueError(f"{expert_name} 的 AHP 判断矩阵维度应为 {n}×{n}，实际为 {matrix.shape}。")
    if np.any(matrix <= 0):
        raise ValueError(f"{expert_name} 的 AHP 判断矩阵必须为正矩阵。")
    if not np.allclose(np.diag(matrix), np.ones(n), atol=1e-10):
        raise ValueError(f"{expert_name} 的 AHP 判断矩阵对角线必须全部为 1。")
    if not np.allclose(matrix * matrix.T, np.ones((n, n)), atol=1e-10):
        raise ValueError(f"{expert_name} 的 AHP 判断矩阵不满足互反性。")


def _ahp_consistency_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(config.RISK_THEME_ORDER)
    n = len(labels)
    expert_matrices = {
        expert_name: np.array(matrix_values, dtype=float)
        for expert_name, matrix_values in config.AHP_EXPERT_PAIRWISE_MATRICES.items()
    }
    if len(expert_matrices) != 3:
        raise ValueError("AHP 专家判断矩阵数量应为 3 个。")

    consistency_rows = []
    matrix_stack = []
    for expert_name, matrix in expert_matrices.items():
        _validate_ahp_matrix(expert_name, matrix, n)
        matrix_stack.append(matrix)
        weights, lambda_max, consistency_index, random_index, consistency_ratio = _calculate_ahp_metrics(matrix)
        consistency_rows.append(
            {
                "matrix_source": expert_name,
                "n_criteria": n,
                "lambda_max": lambda_max,
                "consistency_index": consistency_index,
                "random_index": random_index,
                "consistency_ratio": consistency_ratio,
                "pass_rule": "CR < 0.10",
                "passed": consistency_ratio < 0.10,
            }
        )

    group_matrix = np.exp(np.mean(np.log(np.stack(matrix_stack)), axis=0))
    _validate_ahp_matrix("group_geometric_mean", group_matrix, n)
    weights, lambda_max, consistency_index, random_index, consistency_ratio = _calculate_ahp_metrics(group_matrix)
    consistency_rows.append(
        {
            "matrix_source": "group_geometric_mean",
            "n_criteria": n,
            "lambda_max": lambda_max,
            "consistency_index": consistency_index,
            "random_index": random_index,
            "consistency_ratio": consistency_ratio,
            "pass_rule": "CR < 0.10",
            "passed": consistency_ratio < 0.10,
        }
    )
    weight_df = pd.DataFrame(
        {
            "label": labels,
            "weight": weights,
            "weight_percent": weights * 100,
            "rank": pd.Series(weights).rank(method="first", ascending=False).astype(int),
        }
    ).sort_values("rank")
    consistency_df = pd.DataFrame(consistency_rows)
    return weight_df, consistency_df


def high_score_high_risk_analysis(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    question_meta_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    quadrant_order = ["低分-低风险", "低分-高风险", "高分-低风险", "高分-高风险"]
    quality_order = ["低分", "高分"]
    risk_order = ["低风险", "高风险"]

    risk_df = (
        label_df[label_df["label_type"] == "risk"]
        .drop_duplicates(subset=["model_under_evaluation", "question_id", "label"])
        .copy()
    )
    (
        ahp_weight_df,
        ahp_consistency_df,
    ) = _ahp_consistency_tables()
    risk_weight_map = ahp_weight_df.set_index("label")["weight"].to_dict()

    missing_weights = sorted(set(risk_df["label"]) - set(risk_weight_map))
    if missing_weights:
        raise ValueError(f"以下风险主题缺少 AHP 权重：{missing_weights}")

    risk_count_df = (
        risk_df.groupby(
            ["model_under_evaluation", "model_label", "question_id", "question_text"],
            as_index=False,
        )
        .agg(risk_n=("label", "count"))
    )
    risk_score_df = risk_df.assign(risk_weight=risk_df["label"].map(risk_weight_map))
    risk_score_df = (
        risk_score_df.groupby(
            ["model_under_evaluation", "model_label", "question_id", "question_text"],
            as_index=False,
        )
        .agg(weighted_risk_score=("risk_weight", "sum"))
    )
    risk_label_text_df = (
        risk_df.groupby(
            ["model_under_evaluation", "question_id", "question_text"],
            as_index=False,
        )
        .agg(风险标签=("label", lambda values: "；".join(sorted(values))))
    )

    merged = score_df.merge(
        risk_count_df,
        on=["model_under_evaluation", "model_label", "question_id", "question_text"],
        how="left",
    )
    merged = merged.merge(
        risk_score_df,
        on=["model_under_evaluation", "model_label", "question_id", "question_text"],
        how="left",
    )
    merged["risk_n"] = merged["risk_n"].fillna(0).astype(int)
    merged["weighted_risk_score"] = merged["weighted_risk_score"].fillna(0.0)
    merged["strengths"] = merged["strengths"].apply(lambda values: "；".join(values))
    merged["risks"] = merged["risks"].apply(lambda values: "；".join(values))
    merged = merged.merge(question_meta_df, on=["question_id", "question_text"], how="left")

    quality_cutoff = float(merged["total"].median())
    risk_cutoff = float(merged["weighted_risk_score"].quantile(0.75))
    merged["quality_level"] = pd.Categorical(
        merged["total"].apply(lambda value: "高分" if value >= quality_cutoff else "低分"),
        categories=quality_order,
        ordered=True,
    )
    merged["risk_level"] = pd.Categorical(
        merged["weighted_risk_score"].apply(lambda value: "高风险" if value >= risk_cutoff else "低风险"),
        categories=risk_order,
        ordered=True,
    )
    merged["quadrant"] = pd.Categorical(
        merged["quality_level"].astype(str) + "-" + merged["risk_level"].astype(str),
        categories=quadrant_order,
        ordered=True,
    )
    merged["model_under_evaluation"] = pd.Categorical(
        merged["model_under_evaluation"],
        categories=config.MODEL_ID_ORDER,
        ordered=True,
    )

    threshold_summary = pd.DataFrame(
        [
            {
                "quality_cutoff": quality_cutoff,
                "risk_cutoff": risk_cutoff,
                "quality_rule": f"total >= {quality_cutoff:.1f} 判定为高分",
                "risk_rule": f"weighted_risk_score >= P75({risk_cutoff:.3f}) 判定为高风险",
            }
        ]
    )

    risk_burden_summary = summarize_by_model(
        merged.rename(columns={"weighted_risk_score": "value"}),
        value_col="value",
    )

    quadrant_summary = (
        merged.groupby("quadrant", observed=True)
        .agg(回答数=("question_id", "count"))
        .reindex(quadrant_order, fill_value=0)
        .reset_index()
    )
    quadrant_summary["占比"] = quadrant_summary["回答数"] / len(merged)

    model_index = pd.MultiIndex.from_product(
        [config.MODEL_ID_ORDER, quadrant_order],
        names=["model_under_evaluation", "quadrant"],
    )
    quadrant_by_model = (
        merged.groupby(["model_under_evaluation", "model_label", "quadrant"], observed=True)
        .agg(回答数=("question_id", "count"))
        .reset_index()
        .set_index(["model_under_evaluation", "quadrant"])
        .reindex(model_index, fill_value=0)
        .reset_index()
    )
    quadrant_by_model["model_label"] = quadrant_by_model["model_under_evaluation"].map(config.MODEL_LABEL_MAP)
    quadrant_by_model["占该模型回答比例"] = quadrant_by_model["回答数"] / len(config.QUESTION_META)
    quadrant_by_model["model_under_evaluation"] = pd.Categorical(
        quadrant_by_model["model_under_evaluation"],
        categories=config.MODEL_ID_ORDER,
        ordered=True,
    )
    quadrant_by_model = quadrant_by_model.sort_values(
        ["model_under_evaluation", "quadrant"]
    ).reset_index(drop=True)

    high_score_high_risk_cases = (
        merged[merged["quadrant"] == "高分-高风险"]
        .merge(
            risk_label_text_df,
            on=["model_under_evaluation", "question_id", "question_text"],
            how="left",
        )
        .sort_values(["weighted_risk_score", "total", "paper_question_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    high_score_high_risk_keys = high_score_high_risk_cases[
        ["model_under_evaluation", "question_id", "question_text"]
    ].drop_duplicates()
    high_score_high_risk_label_summary = (
        risk_df.merge(
            high_score_high_risk_keys,
            on=["model_under_evaluation", "question_id", "question_text"],
            how="inner",
        )
        .groupby("label", as_index=False)
        .agg(出现频数=("question_id", "count"))
        .sort_values(["出现频数", "label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if not high_score_high_risk_label_summary.empty:
        high_score_high_risk_label_summary["占高分高风险回答比例"] = (
            high_score_high_risk_label_summary["出现频数"] / len(high_score_high_risk_cases)
        )

    return {
        "risk_ahp_weights": ahp_weight_df,
        "risk_ahp_consistency": ahp_consistency_df,
        "risk_threshold_summary": threshold_summary,
        "risk_burden_summary": risk_burden_summary,
        "risk_quadrant_summary": quadrant_summary,
        "risk_quadrant_by_model": quadrant_by_model,
        "high_score_high_risk_cases": high_score_high_risk_cases,
        "high_score_high_risk_label_summary": high_score_high_risk_label_summary,
    }


def readability_analysis(
    readability_df: pd.DataFrame,
    question_meta_df: pd.DataFrame,
    score_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    summary = summarize_by_model(readability_df, "R")
    wide_df, friedman_df = friedman_test(readability_df, index_col="question_id", value_col="R")
    pairwise_df = pairwise_wilcoxon(wide_df)

    corr_df = readability_df.merge(
        score_df[["model_under_evaluation", "question_id", "total"]],
        on=["model_under_evaluation", "question_id"],
        how="inner",
    )
    correlation_summary = spearman_table(corr_df, x_col="R", y_col="total")
    return {
        "readability_summary": summary,
        "readability_friedman": friedman_df,
        "readability_pairwise_wilcoxon": pairwise_df,
        "readability_quality_correlation": correlation_summary,
    }

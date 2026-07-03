from itertools import combinations

import pandas as pd
from scipy import stats

import config


def _pivot_paired_data(
    df: pd.DataFrame,
    index_col: str,
    value_col: str,
    column_col: str = "model_under_evaluation",
) -> pd.DataFrame:
    wide = (
        df.pivot(index=index_col, columns=column_col, values=value_col)
        .reindex(columns=config.MODEL_ID_ORDER)
        .dropna()
    )
    if wide.empty:
        raise ValueError(f"{value_col} 的配对数据透视后为空。")
    return wide


def friedman_test(df: pd.DataFrame, index_col: str, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = _pivot_paired_data(df, index_col=index_col, value_col=value_col)
    if wide.nunique(axis=1).eq(1).all():
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = stats.friedmanchisquare(*(wide[col].values for col in wide.columns))
    result_df = pd.DataFrame(
        [
            {
                "index_col": index_col,
                "value_col": value_col,
                "n_blocks": len(wide),
                "statistic": statistic,
                "p_value": p_value,
            }
        ]
    )
    return wide, result_df


def pairwise_wilcoxon(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_pairs = list(combinations(config.MODEL_ID_ORDER, 2))
    for model_a, model_b in model_pairs:
        paired = wide_df[[model_a, model_b]].dropna()
        if paired.empty:
            raise ValueError(f"{model_a} 与 {model_b} 的 Wilcoxon 配对样本为空。")
        differences = paired[model_a].values - paired[model_b].values
        if (differences == 0).all():
            statistic, p_value = 0.0, 1.0
            rank_biserial = 0.0
        else:
            statistic, p_value = stats.wilcoxon(
                paired[model_a].values,
                paired[model_b].values,
                alternative="two-sided",
                method="auto",
            )
            nonzero_diff = differences[differences != 0]
            ranks = stats.rankdata(abs(nonzero_diff))
            positive_rank_sum = ranks[nonzero_diff > 0].sum()
            negative_rank_sum = ranks[nonzero_diff < 0].sum()
            rank_biserial = (positive_rank_sum - negative_rank_sum) / (positive_rank_sum + negative_rank_sum)
        rows.append(
            {
                "model_a": config.MODEL_LABEL_MAP[model_a],
                "model_b": config.MODEL_LABEL_MAP[model_b],
                "n_pairs": len(paired),
                "statistic": statistic,
                "p_value": p_value,
                "rank_biserial": rank_biserial,
            }
        )
    result_df = pd.DataFrame(rows)
    result_df["p_value_bonferroni"] = (result_df["p_value"] * len(model_pairs)).clip(upper=1.0)
    return result_df


def summarize_by_model(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(["model_under_evaluation", "model_label"], observed=True)
        .agg(
            均值=(value_col, "mean"),
            标准差=(value_col, "std"),
            中位数=(value_col, "median"),
            最小值=(value_col, "min"),
            最大值=(value_col, "max"),
            下四分位数=(value_col, lambda values: values.quantile(0.25)),
            上四分位数=(value_col, lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    summary["model_under_evaluation"] = pd.Categorical(
        summary["model_under_evaluation"],
        categories=config.MODEL_ID_ORDER,
        ordered=True,
    )
    return summary.sort_values("model_under_evaluation").reset_index(drop=True)


def cochran_q_test(binary_wide_df: pd.DataFrame) -> tuple[float, float]:
    matrix = binary_wide_df.astype(int).to_numpy()
    n_rows, n_cols = matrix.shape
    if n_cols < 3:
        raise ValueError("Cochran's Q 检验至少需要 3 个配对条件。")

    column_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    total = column_sums.sum()
    denominator = n_cols * total - (row_sums ** 2).sum()
    if denominator == 0:
        return float("nan"), float("nan")
    statistic = (n_cols - 1) * (n_cols * (column_sums ** 2).sum() - total ** 2) / denominator
    p_value = 1 - stats.chi2.cdf(statistic, df=n_cols - 1)
    return float(statistic), float(p_value)


def spearman_table(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    rows = []

    rho_all, p_all = stats.spearmanr(df[x_col], df[y_col])
    rows.append(
        {
            "group": "整体",
            "n": len(df),
            "spearman_rho": rho_all,
            "p_value": p_all,
        }
    )

    for model_id in config.MODEL_ID_ORDER:
        subset = df[df["model_under_evaluation"] == model_id]
        rho, p_value = stats.spearmanr(subset[x_col], subset[y_col])
        rows.append(
            {
                "group": config.MODEL_LABEL_MAP[model_id],
                "n": len(subset),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)

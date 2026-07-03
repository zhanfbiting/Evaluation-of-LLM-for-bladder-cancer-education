import os
import shutil
import warnings
from pathlib import Path
from textwrap import fill

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns

import config
from analysis_pipeline import (
    high_score_high_risk_analysis,
    mdiscern_item_analysis,
    overall_quality_analysis,
    question_difficulty_analysis,
    readability_analysis,
    risk_theme_analysis,
)
from data_processing import (
    build_label_table,
    build_professional_term_vocabulary,
    build_question_meta,
    build_readability_table,
    load_answer_text_table,
    load_score_table,
    split_segmented_answers,
)
from plotting_utils import save_figure, set_plot_theme, style_axes, style_colorbar


MODEL_MARKERS = {
    "deepseek": "o",
    "doubao": "s",
    "yuanbao": "^",
}
RISK_LABELS = {
    "部分数据、分类或表述可能不准确": "Potentially inaccurate data, classification, or wording",
    "未充分说明副作用、风险或随访细节": "Insufficient adverse-effect, risk, or follow-up details",
    "未充分说明不确定性或证据等级": "Insufficient uncertainty or evidence grading",
    "未充分说明检查局限性或适用条件": "Insufficient explanation of examination limitations or applicability",
    "用词较绝对，可能忽视个体差异": "Overly absolute wording that may overlook individual differences",
    "个体化建议不足": "Insufficient individualization",
    "缺乏证据来源或指南引用": "Lack of evidence sources or guideline citations",
    "未标注信息时效或更新日期": "Lack of timeliness or update statements",
    "未提供额外支持资源或就医渠道": "Lack of additional support resources or care pathways",
}


def _ensure_output_dirs() -> None:
    for path in [config.OUTPUT_DIR, config.STRUCTURED_OUTPUT_DIR]:
        if path.exists():
            shutil.rmtree(path)
    for path in [
        config.MPLCONFIG_DIR,
        config.FIGURE_DIR,
        config.TABLE_DIR,
        config.STRUCTURED_OUTPUT_DIR / "05_question_difficulty_disagreement_analysis" / "figures",
        config.STRUCTURED_OUTPUT_DIR / "05_high_score_high_risk_analysis" / "figures",
        config.STRUCTURED_OUTPUT_DIR / "06_readability_quality_relationship_analysis" / "figures",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _should_save_index(df: pd.DataFrame) -> bool:
    return not isinstance(df.index, pd.RangeIndex) or df.index.name is not None


def _save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(config.TABLE_DIR / f"{name}.csv", index=_should_save_index(df), encoding="utf-8-sig")


def plot_overall_quality(score_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(
        data=score_df,
        x="model_label",
        y="total",
        hue="model_label",
        order=[config.MODEL_LABEL_MAP[item] for item in config.MODEL_ID_ORDER],
        hue_order=[config.MODEL_LABEL_MAP[item] for item in config.MODEL_ID_ORDER],
        palette=[config.PLOT_COLORS[item] for item in config.MODEL_ID_ORDER],
        linewidth=2.5,
        width=0.55,
        fliersize=0,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.stripplot(
        data=score_df,
        x="model_label",
        y="total",
        order=[config.MODEL_LABEL_MAP[item] for item in config.MODEL_ID_ORDER],
        color="#3E3E3E",
        size=9,
        alpha=0.7,
        ax=ax,
    )
    style_axes(ax, title="Overall mDISCERN Scores by Model", xlabel="Model", ylabel="mDISCERN total score")
    save_figure(fig, output_dir, filename)


def plot_mdiscern_items(item_summary_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    plot_df = item_summary_df.copy()
    plot_df["item_code"] = pd.Categorical(
        plot_df["item_code"],
        categories=list(config.QUESTION_ITEM_LABELS.keys()),
        ordered=True,
    )
    plot_df = plot_df.sort_values(["item_code", "model_under_evaluation"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15, 8))
    for model_id in config.MODEL_ID_ORDER:
        subset = plot_df[plot_df["model_under_evaluation"] == model_id]
        ax.plot(
            subset["item_label"],
            subset["均值"],
            label=config.MODEL_LABEL_MAP[model_id],
            color=config.PLOT_COLORS[model_id],
            linestyle=config.PLOT_LINESTYLES[model_id],
            linewidth=config.LINE_WIDTH,
            marker=MODEL_MARKERS[model_id],
            markersize=config.MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=config.PLOT_COLORS[model_id],
            markeredgewidth=3,
        )
    ax.legend(title="Model", frameon=True)
    style_axes(ax, title="mDISCERN Domain Scores by Model", xlabel="mDISCERN domain", ylabel="Mean score")
    ax.set_ylim(0.8, 5.2)
    ax.tick_params(axis="x", rotation=18)
    save_figure(fig, output_dir, filename)


def plot_question_heatmap(heatmap_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 12))
    heatmap = sns.heatmap(
        heatmap_df,
        cmap=sns.light_palette("#74A7CF", as_cmap=True),
        annot=True,
        fmt=".0f",
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "mDISCERN total score"},
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        ax=ax,
    )
    style_axes(ax, title="Question-Level mDISCERN Scores", xlabel="Model", ylabel="Question No.")
    style_colorbar(heatmap.collections[0].colorbar)
    save_figure(fig, output_dir, filename)


def plot_question_difficulty_scatter(question_difficulty_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    plot_df = question_difficulty_df.copy()
    avg_cutoff = plot_df["平均总分"].median()
    disagreement_cutoff = plot_df["分歧极差"].median()

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(
        plot_df["平均总分"],
        plot_df["分歧极差"],
        s=(config.MARKER_SIZE + 3) ** 2,
        color="white",
        edgecolors="#1F4E79",
        linewidths=3,
        alpha=0.95,
    )
    for _, row in plot_df.iterrows():
        ax.text(
            row["平均总分"] + 0.05,
            row["分歧极差"] + 0.02,
            str(int(row["paper_question_id"])),
            fontsize=15,
            fontweight="bold",
            color="#1F1F1F",
        )

    ax.axvline(avg_cutoff, color="#A52A2A", linestyle="--", linewidth=config.LINE_WIDTH)
    ax.axhline(disagreement_cutoff, color="#556B2F", linestyle="-.", linewidth=config.LINE_WIDTH)
    style_axes(
        ax,
        title="Question Difficulty and Between-Model Disagreement",
        xlabel="Mean total score",
        ylabel="Between-model score range",
    )
    save_figure(fig, output_dir, filename)


def plot_readability(readability_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(
        data=readability_df,
        x="model_label",
        y="R",
        hue="model_label",
        order=[config.MODEL_LABEL_MAP[item] for item in config.MODEL_ID_ORDER],
        hue_order=[config.MODEL_LABEL_MAP[item] for item in config.MODEL_ID_ORDER],
        palette=[config.PLOT_COLORS[item] for item in config.MODEL_ID_ORDER],
        linewidth=2.5,
        width=0.55,
        fliersize=0,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.stripplot(
        data=readability_df,
        x="model_label",
        y="R",
        order=[config.MODEL_LABEL_MAP[item] for item in config.MODEL_ID_ORDER],
        color="#3E3E3E",
        size=8.5,
        alpha=0.7,
        ax=ax,
    )
    style_axes(ax, title="Chinese Readability R Values by Model", xlabel="Model", ylabel="R value")
    save_figure(fig, output_dir, filename)


def plot_risk_theme(risk_theme_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    label_order = (
        risk_theme_df.groupby("label", observed=True)["出现频数"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plot_df = risk_theme_df[risk_theme_df["label"].isin(label_order)].copy()
    plot_df["risk_label"] = plot_df["label"].astype(str).map(RISK_LABELS)
    label_order_en = [RISK_LABELS[label] for label in label_order]

    fig, ax = plt.subplots(figsize=(16, 11))
    sns.barplot(
        data=plot_df,
        y="risk_label",
        x="占比",
        hue="model_label",
        order=label_order_en,
        hue_order=[config.MODEL_LABEL_MAP[item] for item in config.MODEL_ID_ORDER],
        palette=[config.PLOT_COLORS[item] for item in config.MODEL_ID_ORDER],
        edgecolor="#1F1F1F",
        linewidth=2,
        errorbar=None,
        ax=ax,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(title="Model", frameon=True)
    style_axes(ax, title="Occurrence of Potential Risk Themes", xlabel="Occurrence proportion", ylabel="Risk theme")
    save_figure(fig, output_dir, filename)


def plot_high_score_high_risk_labels(label_summary_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    if label_summary_df.empty:
        return

    plot_df = label_summary_df.copy().head(6).sort_values("出现频数", ascending=True)
    plot_df["wrapped_label"] = plot_df["label"].map(lambda value: fill(RISK_LABELS.get(str(value), str(value)), width=34))
    colors = ["#B8860B", "#556B2F", "#1F4E79", "#A52A2A", "#4B5563", "#7A3E2E"]

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    bars = ax.barh(
        plot_df["wrapped_label"],
        plot_df["占高分高风险回答比例"],
        color=colors[-len(plot_df) :],
        edgecolor="#1F1F1F",
        linewidth=2,
        height=0.62,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_xlim(0, min(1.18, max(1.05, float(plot_df["占高分高风险回答比例"].max()) + 0.18)))

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        value = float(row["占高分高风险回答比例"])
        ax.text(
            value + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['出现频数'])} ({value:.1%})",
            va="center",
            ha="left",
            fontsize=15,
            fontweight="bold",
            color="#263238",
        )
    ax.grid(axis="x", color="#D7E3E8", linewidth=1.1, alpha=0.9)
    ax.grid(axis="y", visible=False)
    style_axes(ax, title="Main Risk Sources in High-Score / High-WRS Responses", xlabel="Occurrence proportion", ylabel="")
    ax.tick_params(axis="y", length=10, pad=10)
    save_figure(fig, output_dir, filename)


def main() -> None:
    warnings.filterwarnings("ignore", message="Sample size too small for normal approximation.")

    _ensure_output_dirs()
    set_plot_theme()

    score_df = load_score_table()
    answer_text_df = load_answer_text_table()
    question_meta_df = build_question_meta()
    segmented_answer_df = split_segmented_answers(question_meta_df)
    term_vocab, vocabulary_df = build_professional_term_vocabulary()
    label_df = build_label_table(score_df)
    readability_df, calibration_df = build_readability_table(answer_text_df, segmented_answer_df, term_vocab)

    overall_results = overall_quality_analysis(score_df)
    mdiscern_results = mdiscern_item_analysis(score_df)
    question_results = question_difficulty_analysis(score_df, question_meta_df)
    risk_results = risk_theme_analysis(label_df, question_meta_df)
    high_risk_results = high_score_high_risk_analysis(score_df, label_df, question_meta_df)
    readability_results = readability_analysis(readability_df, question_meta_df, score_df)

    score_output = score_df.copy()
    score_output["strengths"] = score_output["strengths"].apply(lambda values: "；".join(values))
    score_output["risks"] = score_output["risks"].apply(lambda values: "；".join(values))
    segmented_output = segmented_answer_df.copy()
    segmented_output["segmented_tokens"] = segmented_output["segmented_tokens"].apply(lambda values: " / ".join(values))

    _save_tables(
        {
            "01_score_master": score_output,
            "02_answer_text": answer_text_df,
            "03_question_list": question_meta_df,
            "04_label_expanded": label_df,
            "05_readability": readability_df,
            "06_readability_calibration": calibration_df,
            "07_term_vocabulary": vocabulary_df,
            "08_segmented_answers": segmented_output,
            **overall_results,
            **mdiscern_results,
            **question_results,
            **risk_results,
            **high_risk_results,
            **readability_results,
        }
    )

    plot_overall_quality(score_df, config.FIGURE_DIR, "02_overall_information_quality")
    plot_mdiscern_items(mdiscern_results["mdiscern_item_summary"], config.FIGURE_DIR, "03_mdiscern_domain_comparison")
    plot_question_heatmap(question_results["question_score_heatmap"], config.FIGURE_DIR, "05_question_difficulty_disagreement_heatmap")
    plot_question_difficulty_scatter(
        question_results["question_difficulty_summary"],
        config.STRUCTURED_OUTPUT_DIR / "05_question_difficulty_disagreement_analysis" / "figures",
        "05_question_difficulty_disagreement_quadrant",
    )
    plot_readability(
        readability_df,
        config.STRUCTURED_OUTPUT_DIR / "06_readability_quality_relationship_analysis" / "figures",
        "06_readability_r_value_boxplot",
    )
    plot_risk_theme(risk_results["risk_theme_summary"], config.FIGURE_DIR, "06_risk_theme_comparison")
    plot_high_score_high_risk_labels(
        high_risk_results["high_score_high_risk_label_summary"],
        config.STRUCTURED_OUTPUT_DIR / "05_high_score_high_risk_analysis" / "figures",
        "05_high_score_high_risk_risk_sources",
    )

    print("Analysis complete.")
    print(f"Tables: {config.TABLE_DIR}")
    print(f"Figures: {config.FIGURE_DIR} and selected structured figure folders")


if __name__ == "__main__":
    main()

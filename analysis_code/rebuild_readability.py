"""Rebuild readability only; keep mDISCERN, guideline and risk scores unchanged."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data_processing import (
    build_question_meta, load_answer_text_table, split_segmented_answers,
    build_professional_term_vocabulary, build_readability_table, load_score_table,
)
from analysis_pipeline import readability_analysis


def main():
    raw = load_answer_text_table()
    meta = build_question_meta()
    segmented = split_segmented_answers(meta)
    vocabulary, vocab_table = build_professional_term_vocabulary()
    values, audit = build_readability_table(raw, segmented, vocabulary)
    scores = load_score_table()
    assert len(values) == 60
    assert set(values.model_under_evaluation) == set(config.MODEL_ID_ORDER)
    assert not values.duplicated(["model_under_evaluation", "question_id"]).any()
    assert values.groupby("model_under_evaluation").size().eq(20).all()
    assert values.professional_term_ratio.between(0, 1).all()
    expected = 17.5255 + .0024 * values.total_chars + .04415 * values.avg_sentence_length - 18.3344 * (1 - values.professional_term_ratio)
    assert np.allclose(values.R, expected)
    results = readability_analysis(values, meta, scores)
    components = values.groupby(["model_under_evaluation", "model_label"]).agg(
        chars_mean=("total_chars", "mean"), chars_sd=("total_chars", "std"),
        sentences_mean=("sentence_count", "mean"), sentences_sd=("sentence_count", "std"),
        sentence_length_mean=("avg_sentence_length", "mean"), sentence_length_sd=("avg_sentence_length", "std"),
        term_ratio_mean=("professional_term_ratio", "mean"), term_ratio_sd=("professional_term_ratio", "std"),
        R_mean=("R", "mean"), R_sd=("R", "std"),
    ).reset_index()
    segmented_out = segmented.drop(columns=["segmented_tokens"]).copy()
    tables = {
        "02_answer_text": raw, "05_readability": values,
        "06_readability_input_audit": audit,
        "07_term_vocabulary": vocab_table,
        "08_segmented_answers": segmented_out,
        "readability_components_summary": components,
        **results,
    }
    for name, frame in tables.items():
        config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(config.TABLE_DIR / (name + ".csv"), index=False, encoding="utf-8-sig")
    # Remove only the obsolete calibration output, not any source record.
    obsolete = config.TABLE_DIR / "06_readability_calibration.csv"
    if obsolete.exists():
        obsolete.unlink()

    figure_dir = config.STRUCTURED_OUTPUT_DIR / "readability" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    order = [config.MODEL_LABEL_MAP[x] for x in config.MODEL_ID_ORDER]
    groups = [values[values.model_label == x].R.to_numpy() for x in order]
    bp = ax.boxplot(groups, tick_labels=order, patch_artist=True, showfliers=False)
    for box, model in zip(bp["boxes"], config.MODEL_ID_ORDER):
        box.set_facecolor(config.PLOT_COLORS[model])
        box.set_alpha(.3)
    rng = np.random.default_rng(20261009)
    for i, group in enumerate(groups, 1):
        ax.scatter(i + rng.uniform(-.1, .1, len(group)), group, s=24, color="#333333", alpha=.75)
    ax.set_xlabel("Product")
    ax.set_ylabel("Chinese readability R value")
    ax.set_title("Chinese Readability R Values by Product")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(figure_dir / ("readability_r_values." + ext), dpi=300)
    plt.close(fig)
    evidence = {
        "raw_response_files": {k: {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in config.MODEL_DOCX_MAP.items()},
        "glossary_sha256": hashlib.sha256(config.GLOSSARY_TXT.read_bytes()).hexdigest(),
        "glossary_unique_terms": len(vocabulary),
        "n_responses": len(values), "calibration_applied": False,
        "characters": "letters and digits only; punctuation, symbols and whitespace excluded",
        "sentences": "nonempty segments delimited by full stops/question/exclamation marks; numeric decimal/list dots excluded",
        "medical_terms": "shared glossary, longest non-overlapping character matches; no product-specific adjustment",
        "summary": components.to_dict("records"),
        "friedman": results["readability_friedman"].to_dict("records"),
        "pairwise": results["readability_pairwise_wilcoxon"].to_dict("records"),
        "correlation": results["readability_quality_correlation"].to_dict("records"),
    }
    (config.TABLE_DIR / "readability_rebuild_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

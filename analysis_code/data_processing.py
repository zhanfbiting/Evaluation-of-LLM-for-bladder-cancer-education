from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd

import config


SENTENCE_SPLIT_REGEX = re.compile(r"[。！？；!?;]+")
TOKEN_SEPARATOR = "/"
QUESTION_SEPARATOR_PATTERN = r'[\/\s，,。！？!?；;：:"“”‘’（）()、]*'
PURE_PUNCTUATION_REGEX = re.compile(r"^[\W_]+$", re.UNICODE)


def _clean_loaded_answer_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n[ \t]*[•●]\s*", "\n", text)
    text = re.sub(r"^[ \t]*[•●]\s*", "", text, flags=re.M)
    text = text.replace("\t•\t", "\n")
    text = text.replace("\t•", "\n")
    text = text.replace("•\t", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    slash_count = text.count("/")
    if slash_count > 200 and slash_count > max(1, len(text) // 20):
        text = text.replace("/", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_answer_docx_as_text(path: Path) -> str:
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return _clean_loaded_answer_text(result.stdout)


def _question_body_to_regex(question: str) -> str:
    body = re.escape(question.strip())
    body = body.replace(r"\？", r"[？?]?")
    body = body.replace(r"\?", r"[？?]?")
    if not question.strip().endswith(("？", "?")):
        body = body + r"[？?]?"
    return body


def _find_answer_heading(text: str, question_id: int, question_text: str) -> re.Match[str] | None:
    body = _question_body_to_regex(question_text)
    patterns = [
        rf"(?m)^[\s\t\"“”'‘’]*{body}\s*$",
        rf"(?m)^[\s\t\"“”'‘’]*(?:{question_id}\s*[\.、]?\s*){body}\s*$",
        rf"(?m)^[\s\t\"“”'‘’]*(?:{question_id}\s*[\.、]?\s*)?{body}\s*$",
    ]
    best_match = None
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        if best_match is None or match.start() < best_match.start():
            best_match = match
    return best_match


def _extract_raw_answer_rows(question_meta: pd.DataFrame) -> list[dict]:
    rows = []
    question_records = question_meta.sort_values("question_id").to_dict(orient="records")
    for model_id in config.MODEL_ID_ORDER:
        source_path = config.MODEL_DOCX_MAP[model_id]
        full_text = _read_answer_docx_as_text(source_path)
        headings = []
        for record in question_records:
            match = _find_answer_heading(full_text, record["question_id"], record["question_text"])
            if match is None:
                raise ValueError(
                    f"模型 `{model_id}` 的原始回答文档中，问题 {record['question_id']} 标题未匹配到。"
                )
            headings.append(
                {
                    "question_id": record["question_id"],
                    "question_text": record["question_text"],
                    "start": match.start(),
                    "answer_start": match.end(),
                }
            )
        headings = sorted(headings, key=lambda item: item["start"])
        for index, item in enumerate(headings):
            next_start = headings[index + 1]["start"] if index + 1 < len(headings) else len(full_text)
            answer_text = full_text[item["answer_start"] : next_start].strip()
            answer_text = answer_text.strip("\"“”' \n\t")
            answer_text = re.sub(r"\n{3,}", "\n\n", answer_text)
            if not answer_text:
                raise ValueError(f"模型 `{model_id}` 的问题 {item['question_id']} 回答为空。")
            rows.append(
                {
                    "model_under_evaluation": model_id,
                    "model_label": config.MODEL_LABEL_MAP[model_id],
                    "question_id": item["question_id"],
                    "question_text": item["question_text"],
                    "answer_text": answer_text,
                }
            )
    return rows


def load_score_table() -> pd.DataFrame:
    rows = []
    with config.SCORE_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            result = payload["result"]
            row = {
                "model_under_evaluation": payload["model_under_evaluation"],
                "model_label": config.MODEL_LABEL_MAP[payload["model_under_evaluation"]],
                "question_id": payload["question_id"],
                "question_text": payload["question_text"],
                "one_sentence_summary": result["one_sentence_summary"],
                "strengths": result["strengths"],
                "risks": result["risks"],
            }
            for key in [f"q{i}" for i in range(1, 9)] + ["total"]:
                row[key] = result[key]
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["question_id", "model_under_evaluation"]).reset_index(drop=True)
    expected_rows = len(config.MODEL_ID_ORDER) * len(config.QUESTION_META)
    if len(df) != expected_rows:
        raise ValueError(f"评分主表行数异常，期望 {expected_rows} 行，实际 {len(df)} 行。")
    return df


def load_answer_text_table() -> pd.DataFrame:
    question_meta = build_question_meta()
    rows = _extract_raw_answer_rows(question_meta)
    df = pd.DataFrame(rows).sort_values(["question_id", "model_under_evaluation"]).reset_index(drop=True)
    expected_rows = len(config.MODEL_ID_ORDER) * len(config.QUESTION_META)
    if len(df) != expected_rows:
        raise ValueError(f"回答文本表行数异常，期望 {expected_rows} 行，实际 {len(df)} 行。")
    return df


def build_question_meta() -> pd.DataFrame:
    df = pd.DataFrame(
        config.QUESTION_META,
        columns=["question_id", "paper_question_id", "question_text", "question_short"],
    )
    return df.sort_values("paper_question_id").reset_index(drop=True)


def _load_mapping_table(path: Path) -> tuple[dict[str, list[str]], set[str]]:
    mapping: dict[str, list[str]] = {}
    standardized_options: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_label = row[reader.fieldnames[0]].strip()
            targets = [item.strip() for item in row[reader.fieldnames[1]].split("；") if item.strip()]
            mapping[raw_label] = targets
            standardized_options.update(targets)
    return mapping, standardized_options


def _expand_labels(
    score_df: pd.DataFrame,
    label_column: str,
    mapping_path: Path,
    label_type: str,
) -> pd.DataFrame:
    mapping, standardized_options = _load_mapping_table(mapping_path)
    rows = []
    for record in score_df.to_dict(orient="records"):
        labels = record[label_column]
        for raw_label in labels:
            if raw_label in mapping:
                standardized_labels = mapping[raw_label]
            elif raw_label in standardized_options:
                standardized_labels = [raw_label]
            else:
                raise ValueError(f"{label_type} 标签 `{raw_label}` 未在映射表中找到。")
            for standardized_label in standardized_labels:
                rows.append(
                    {
                        "model_under_evaluation": record["model_under_evaluation"],
                        "model_label": record["model_label"],
                        "question_id": record["question_id"],
                        "question_text": record["question_text"],
                        "label_type": label_type,
                        "raw_label": raw_label,
                        "label": standardized_label,
                    }
                )
    return pd.DataFrame(rows)


def build_label_table(score_df: pd.DataFrame) -> pd.DataFrame:
    strength_df = _expand_labels(score_df, "strengths", config.STRENGTHS_MAPPING_CSV, "strength")
    risk_df = _expand_labels(score_df, "risks", config.RISKS_MAPPING_CSV, "risk")
    label_df = pd.concat([strength_df, risk_df], ignore_index=True)
    return label_df.sort_values(["label_type", "question_id", "model_under_evaluation", "label"]).reset_index(drop=True)


def _parse_term_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        second_line = handle.readline().strip()
        third_line = handle.readline().strip()
        handle.readline()

        total_match = re.search(r"(\d+)", first_line)
        token_match = re.search(r"(\d+)", second_line)
        ratio_match = re.search(r"([0-9.]+)%", third_line)
        if not (total_match and token_match and ratio_match):
            raise ValueError(f"术语汇总文件 `{path}` 头部格式异常。")

        term_counts: dict[str, int] = {}
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("-") or not stripped:
                continue
            if ": " not in stripped or not stripped.endswith("次"):
                continue
            term_label, rest = stripped.rsplit(": ", 1)
            count_text = rest[:-1].strip().rstrip("次").strip()
            if not count_text.isdigit():
                continue
            term = term_label.split("\t", 1)[0].strip()
            term_counts[term] = int(count_text)

    return {
        "provided_term_total": int(total_match.group(1)),
        "provided_token_total": int(token_match.group(1)),
        "provided_ratio": float(ratio_match.group(1)) / 100.0,
        "term_counts": term_counts,
    }


def load_glossary_terms() -> set[str]:
    terms: set[str] = set()
    with config.GLOSSARY_TXT.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            term = line.split("\t", 1)[0].strip()
            if term:
                terms.add(term)
    return terms


def build_professional_term_vocabulary() -> tuple[set[str], pd.DataFrame]:
    glossary_terms = load_glossary_terms()
    term_rows = []
    active_terms: set[str] = set()
    for model_id in config.MODEL_ID_ORDER:
        parsed = _parse_term_summary(config.MODEL_TERM_SUMMARY_MAP[model_id])
        for term, count in parsed["term_counts"].items():
            if count > 0:
                active_terms.add(term)
            term_rows.append(
                {
                    "model_under_evaluation": model_id,
                    "model_label": config.MODEL_LABEL_MAP[model_id],
                    "term": term,
                    "count": count,
                }
            )
    if not active_terms:
        raise ValueError("未从现有专业术语汇总文件中解析出任何非零术语。")
    missing_terms = sorted(term for term in active_terms if term not in glossary_terms)
    if missing_terms:
        raise ValueError(f"以下术语未包含在词汇表中：{missing_terms[:10]}")
    vocabulary_df = pd.DataFrame(term_rows)
    return active_terms, vocabulary_df


def _read_docx_as_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _build_heading_pattern(question_id: int, question_text: str) -> re.Pattern[str]:
    chars = [ch for ch in question_text if ch not in config.PUNCT_STRIP_CHARS and not ch.isspace()]
    if not chars:
        raise ValueError(f"问题 `{question_text}` 无法构建标题匹配模式。")
    body = QUESTION_SEPARATOR_PATTERN.join(re.escape(ch) for ch in chars)
    prefix = rf"(?:^|\n+)\s*(?:{question_id}[./、/\s]*)?"
    suffix = QUESTION_SEPARATOR_PATTERN + r"(?=\s*(?:\n|$))"
    return re.compile(prefix + body + suffix)


def _contains_content(token: str) -> bool:
    return bool(re.search(r"[0-9A-Za-z\u4e00-\u9fff]", token))


def split_segmented_answers(question_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id in config.MODEL_ID_ORDER:
        full_text = _read_docx_as_text(config.MODEL_DOCX_MAP[model_id])
        matches = []
        for record in question_meta.to_dict(orient="records"):
            pattern = _build_heading_pattern(record["question_id"], record["question_text"])
            found = list(pattern.finditer(full_text))
            if len(found) == 0:
                raise ValueError(
                    f"模型 `{model_id}` 的分词文档中，问题 {record['question_id']} 标题匹配到 0 次。"
                )
            match = found[-1]
            matches.append(
                {
                    "question_id": record["question_id"],
                    "question_text": record["question_text"],
                    "question_short": record["question_short"],
                    "start": match.start(),
                    "answer_start": match.end(),
                    "heading_match_count": len(found),
                }
            )
        matches = sorted(matches, key=lambda item: item["start"])
        for index, item in enumerate(matches):
            next_start = matches[index + 1]["start"] if index + 1 < len(matches) else len(full_text)
            answer_segment = full_text[item["answer_start"] : next_start].strip()
            tokens = [token.strip() for token in answer_segment.split(TOKEN_SEPARATOR) if token.strip()]
            rows.append(
                {
                    "model_under_evaluation": model_id,
                    "model_label": config.MODEL_LABEL_MAP[model_id],
                    "question_id": item["question_id"],
                    "question_text": item["question_text"],
                    "segmented_answer_text": answer_segment,
                    "segmented_token_count": len(tokens),
                    "segmented_tokens": tokens,
                    "heading_match_count": item["heading_match_count"],
                }
            )
    df = pd.DataFrame(rows).sort_values(["question_id", "model_under_evaluation"]).reset_index(drop=True)
    expected_rows = len(config.MODEL_ID_ORDER) * len(config.QUESTION_META)
    if len(df) != expected_rows:
        raise ValueError(f"分词回答表行数异常，期望 {expected_rows} 行，实际 {len(df)} 行。")
    return df


def _count_professional_terms(tokens: list[str], term_vocab: set[str]) -> Counter:
    term_counts: Counter = Counter()
    max_length = max(len(term) for term in term_vocab)
    for start_index, token in enumerate(tokens):
        if not _contains_content(token):
            continue
        current = ""
        for offset in range(start_index, len(tokens)):
            next_token = tokens[offset]
            if not _contains_content(next_token):
                break
            current += next_token
            if len(current) > max_length:
                break
            if current in term_vocab:
                term_counts[current] += 1
    return term_counts


def _character_count(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _sentence_count(text: str) -> int:
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_REGEX.split(text) if piece.strip()]
    return max(len(pieces), 1)


def build_readability_table(
    answer_text_df: pd.DataFrame,
    segmented_answer_df: pd.DataFrame,
    term_vocab: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    readability_rows = []
    calibration_rows = []

    merged = answer_text_df.merge(
        segmented_answer_df,
        on=["model_under_evaluation", "model_label", "question_id", "question_text"],
        how="inner",
    )
    if len(merged) != len(answer_text_df):
        raise ValueError("原始回答文本与分词回答文本未能一一对应。")

    summary_info = {model_id: _parse_term_summary(path) for model_id, path in config.MODEL_TERM_SUMMARY_MAP.items()}
    observed_ratio_by_model: dict[str, float] = {}
    per_row_payloads: dict[str, list[dict]] = {model_id: [] for model_id in config.MODEL_ID_ORDER}

    for record in merged.to_dict(orient="records"):
        term_counter = _count_professional_terms(record["segmented_tokens"], term_vocab)
        professional_term_count = sum(term_counter.values())
        segmented_token_count = record["segmented_token_count"]
        if segmented_token_count == 0:
            raise ValueError(f"问题 {record['question_id']} 的分词 token 数为 0。")
        raw_term_ratio = professional_term_count / segmented_token_count

        payload = {
            "model_under_evaluation": record["model_under_evaluation"],
            "model_label": record["model_label"],
            "question_id": record["question_id"],
            "question_text": record["question_text"],
            "answer_text": record["answer_text"],
            "total_chars": _character_count(record["answer_text"]),
            "sentence_count": _sentence_count(record["answer_text"]),
            "avg_sentence_length": _character_count(record["answer_text"]) / _sentence_count(record["answer_text"]),
            "segmented_token_count": segmented_token_count,
            "professional_term_count_raw": professional_term_count,
            "professional_term_ratio_raw": raw_term_ratio,
        }
        per_row_payloads[record["model_under_evaluation"]].append(payload)

    for model_id, payloads in per_row_payloads.items():
        total_professional_terms = sum(item["professional_term_count_raw"] for item in payloads)
        total_tokens = sum(item["segmented_token_count"] for item in payloads)
        observed_ratio = total_professional_terms / total_tokens
        observed_ratio_by_model[model_id] = observed_ratio
        provided_ratio = summary_info[model_id]["provided_ratio"]
        calibration_factor = provided_ratio / observed_ratio
        calibration_rows.append(
            {
                "model_under_evaluation": model_id,
                "model_label": config.MODEL_LABEL_MAP[model_id],
                "provided_professional_term_total": summary_info[model_id]["provided_term_total"],
                "provided_token_total": summary_info[model_id]["provided_token_total"],
                "provided_ratio": provided_ratio,
                "reconstructed_professional_term_total": total_professional_terms,
                "reconstructed_token_total": total_tokens,
                "reconstructed_ratio": observed_ratio,
                "calibration_factor": calibration_factor,
            }
        )
        for item in payloads:
            calibrated_ratio = item["professional_term_ratio_raw"] * calibration_factor
            readability_rows.append(
                {
                    **item,
                    "professional_term_ratio": calibrated_ratio,
                    "R": 17.5255
                    + 0.0024 * item["total_chars"]
                    + 0.04415 * item["avg_sentence_length"]
                    - 18.3344 * (1 - calibrated_ratio),
                }
            )

    readability_df = pd.DataFrame(readability_rows).sort_values(
        ["question_id", "model_under_evaluation"]
    ).reset_index(drop=True)
    calibration_df = pd.DataFrame(calibration_rows).sort_values("model_under_evaluation").reset_index(drop=True)
    return readability_df, calibration_df


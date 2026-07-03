from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

SUMMARY_DIR = PROJECT_DIR / "source_data" / "model_outputs"

OUTPUT_DIR = CODE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MPLCONFIG_DIR = CODE_DIR / ".mplconfig"
STRUCTURED_OUTPUT_DIR = CODE_DIR / "structured_results"

SCORE_JSONL = PROJECT_DIR / "reviewer_scoring_outputs_clean.jsonl"
GLOSSARY_TXT = SUMMARY_DIR / "medical_glossary.txt"
STRENGTHS_MAPPING_CSV = PROJECT_DIR / "strengths_mapping.csv"
RISKS_MAPPING_CSV = PROJECT_DIR / "risks_mapping.csv"

MODEL_ID_ORDER = ["deepseek", "doubao", "yuanbao"]
MODEL_LABEL_MAP = {
    "deepseek": "DeepSeek",
    "doubao": "Doubao",
    "yuanbao": "Yuanbao",
}
MODEL_DOCX_MAP = {
    "deepseek": SUMMARY_DIR / "deepseek.docx",
    "doubao": SUMMARY_DIR / "doubao.docx",
    "yuanbao": SUMMARY_DIR / "yuanbao.docx",
}
MODEL_TERM_SUMMARY_MAP = {
    "deepseek": SUMMARY_DIR / "deepseek_term_summary.txt",
    "doubao": SUMMARY_DIR / "doubao_term_summary.txt",
    "yuanbao": SUMMARY_DIR / "yuanbao_term_summary.txt",
}

QUESTION_META = [
    (2, 1, "血尿就是膀胱癌吗", "血尿鉴别"),
    (4, 2, "男性膀胱癌症状的早期表现", "早期症状"),
    (12, 3, "膀胱癌除了血尿，还会有尿频尿急尿痛、夜尿增多这些症状吗？", "其他症状"),
    (18, 4, "膀胱癌患者尿血有血块，是不是很严重？怎么快速止血？", "血块止血"),
    (1, 5, "得了膀胱癌能活多久", "生存期"),
    (5, 6, "膀胱癌复发率高吗？怎么预防复发？", "复发预防"),
    (9, 7, "膀胱癌容易骨转移吗", "骨转移"),
    (14, 8, "高级别浸润性膀胱癌患者，没转移的情况下能活几年？", "高级别预后"),
    (15, 9, "膀胱癌电切术后会复发吗？复发几率大概是多少？", "电切复发"),
    (6, 10, "膀胱癌术后灌注治疗有什么副作用？要注意什么？", "灌注副作用"),
    (7, 11, "70多岁老人膀胱癌适合手术吗", "老年手术"),
    (8, 12, "70多岁膀胱癌怎么治疗方案", "老年方案"),
    (10, 13, "膀胱癌有哪些靶向药？", "靶向药"),
    (11, 14, "卡介苗能治疗膀胱癌吗？", "卡介苗"),
    (13, 15, "浸润性膀胱癌必须做膀胱全切手术吗？有没有保守治疗方案？", "保膀胱方案"),
    (3, 16, "膀胱癌患者有哪些饮食禁忌？", "饮食禁忌"),
    (16, 17, "男性膀胱癌患者术后，对夫妻生活、生育有影响吗？", "性功能生育"),
    (17, 18, "膀胱癌治疗要花多少钱？", "治疗费用"),
    (19, 19, "男性膀胱癌复查膀胱镜需要住院吗", "膀胱镜住院"),
    (20, 20, "男性膀胱癌术后家庭换尿袋方法", "换尿袋"),
]

RISK_THEME_ORDER = [
    "部分数据、分类或表述可能不准确",
    "未充分说明副作用、风险或随访细节",
    "未充分说明不确定性或证据等级",
    "未充分说明检查局限性或适用条件",
    "用词较绝对，可能忽视个体差异",
    "个体化建议不足",
    "缺乏证据来源或指南引用",
    "未标注信息时效或更新日期",
    "未提供额外支持资源或就医渠道",
]

AHP_EXPERT_PAIRWISE_MATRICES = {
    "expert_1": [
        [1, 1, 1, 2, 2, 2, 2, 4, 7],
        [1, 1, 1, 1, 2, 2, 2, 3, 6],
        [1, 1, 1, 1, 1, 1, 2, 3, 5],
        [1 / 2, 1, 1, 1, 1, 1, 1, 2, 4],
        [1 / 2, 1 / 2, 1, 1, 1, 1, 1, 2, 3],
        [1 / 2, 1 / 2, 1, 1, 1, 1, 1, 2, 3],
        [1 / 2, 1 / 2, 1 / 2, 1, 1, 1, 1, 2, 3],
        [1 / 4, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 2],
        [1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 3, 1 / 3, 1 / 2, 1],
    ],
    "expert_2": [
        [1, 1, 1, 1, 2, 2, 2, 4, 6],
        [1, 1, 1, 1, 2, 2, 2, 4, 6],
        [1, 1, 1, 1, 1, 2, 2, 3, 5],
        [1, 1, 1, 1, 1, 1, 1, 2, 4],
        [1 / 2, 1 / 2, 1, 1, 1, 1, 1, 2, 4],
        [1 / 2, 1 / 2, 1 / 2, 1, 1, 1, 1, 2, 3],
        [1 / 2, 1 / 2, 1 / 2, 1, 1, 1, 1, 2, 3],
        [1 / 4, 1 / 4, 1 / 3, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 2],
        [1 / 6, 1 / 6, 1 / 5, 1 / 4, 1 / 4, 1 / 3, 1 / 3, 1 / 2, 1],
    ],
    "expert_3": [
        [1, 1, 1, 2, 2, 2, 3, 4, 7],
        [1, 1, 1, 1, 2, 1, 2, 3, 5],
        [1, 1, 1, 1, 1, 1, 2, 3, 5],
        [1 / 2, 1, 1, 1, 1, 1, 1, 2, 4],
        [1 / 2, 1 / 2, 1, 1, 1, 1, 1, 2, 3],
        [1 / 2, 1, 1, 1, 1, 1, 1, 2, 4],
        [1 / 3, 1 / 2, 1 / 2, 1, 1, 1, 1, 2, 3],
        [1 / 4, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 2],
        [1 / 7, 1 / 5, 1 / 5, 1 / 4, 1 / 3, 1 / 4, 1 / 3, 1 / 2, 1],
    ],
}

QUESTION_ITEM_LABELS = {
    "q1": "Clear aims",
    "q2": "Achieves aims",
    "q3": "Relevance",
    "q4": "Sources",
    "q5": "Timeliness",
    "q6": "Balance",
    "q7": "Support resources",
    "q8": "Uncertainty",
}

PUNCT_STRIP_CHARS = set('？?，,。！!；;：:"“”‘’（）()、')

PLOT_COLORS = {
    "deepseek": "#1F4E79",
    "doubao": "#A52A2A",
    "yuanbao": "#556B2F",
}
PLOT_LINESTYLES = {
    "deepseek": "-",
    "doubao": "--",
    "yuanbao": "-.",
}

TITLE_SIZE = 24
LABEL_SIZE = 22
TICK_SIZE = 20
LEGEND_SIZE = 20
LINE_WIDTH = 5
MARKER_SIZE = 13

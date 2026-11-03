

## Run

```bash
python3 run_analysis.py
```

This command writes the complete mDISCERN, risk-theme, AHP/WRS and readability
outputs to `outputs/tables/` and `outputs/figures/`. Risk-theme outputs include
Cochran's Q tests and all three paired exact McNemar comparisons with
Bonferroni-adjusted P values.

To reproduce the guideline-related response-level subtotal, evidence-
source/timeliness, product summaries and paired comparisons, run:

```bash
python3 guideline_concordance_analysis.py
```

## Readability

Run `python rebuild_readability.py` from this directory to reproduce only the
readability outputs and figure. Install `requirements.txt` first. Input paths
are relative to this package and are declared in `config.py`.

The complete original response DOCX files provide the analysis text. `MODEL_DOCX_MAP`
identifies the three original inputs used by the readability workflow.

Response character counts exclude punctuation, symbols and whitespace.
Sentence counts use full stops/question/exclamation marks and exclude numeric
decimal/list dots. Medical-term character coverage uses a shared glossary and
longest non-overlapping matches. T is medical-term characters / total counted
characters. No product-specific calibration or old summary totals enter R.

Outputs include response-level measurements, summary statistics, Friedman
tests, Bonferroni pairwise tests and correlations. All response-level
measurements are calculated directly from the complete original responses.

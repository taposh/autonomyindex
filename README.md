# Autonomy Index (AIx)

Instrument, data, scoring pipeline, and analysis code for:

> **Auditing the Human–LLM Autonomy Gap in Clinical Ethics: Development and Application of the Autonomy Index Across 50 Clinical Ethics Vignettes**
> Taposh Dutta Roy, MS, MBA, MBE; Rebecca Weintraub Brendel, MD, JD
> Center for Bioethics, Harvard Medical School, Boston, MA, USA
> Preprint: medRxiv (Medical Ethics). Manuscript draft in this repo: `Autonomy_Index_medRxiv_v30.docx`

Respect for autonomy is central to biomedical ethics but is rarely assessed with a structured,
reproducible measure. The **Autonomy Index (AIx)** is a 13-item instrument that characterizes
autonomous agency in clinical-ethics vignettes. This repo contains the instrument, the rated
vignette corpus, the LLM scoring pipeline, and the three analysis notebooks that produce every
figure and table in the paper.

---

## Headline result

Across 50 clinical-ethics vignettes, **173 human ratings** (30 clinical and bioethics-informed
reviewers) were compared with **3,431 valid model scores** from **8 general-purpose LLMs** (10 runs
each, July 2026).

| | Human | LLM | Δ (H−L) |
|---|---|---|---|
| **Autonomy Index** | 50.20 | 39.19 | **+11.01** |
| Value Awareness | 54.67 | 44.25 | +10.42 |
| Factual Understanding | 47.74 | 34.03 | +13.71 |
| Rational Deliberation | 47.89 | 39.97 | +7.92 |
| Intentional Action | 50.80 | 38.52 | +12.28 |

- Human and model composites are **strongly associated** (Pearson r = 0.781, P = 2.3 × 10⁻¹¹;
  Spearman ρ = 0.775) — the two sources rank cases similarly.
- The paired human-minus-model difference is **11.0 index points** (95% CI 6.3–15.8; P = 2.6 × 10⁻⁵;
  d_z = 0.66). Humans scored higher on 36 of 50 vignettes.
- **Limits of agreement are wide** (−21.9 to 43.9): the mean gap is not a prediction for any
  individual case.
- **Vignette content explains 80.9%** of score variance; rater source explains **0.7%**.
- Models **never** selected "not applicable"; human reviewers did so on **22.5%** of item responses.
- The domain-by-source joint test found **no heterogeneity** across the four domains
  (Wald χ²[3] = 2.27, P = 0.52) — the gap is uniform, not domain-specific.

Per-model gap vs. humans (mixed model, index points; all P < 0.001):

| Model | Gap | 95% CI |
|---|---|---|
| Gemini 3.5 Flash Lite | −6.95 | −10.65, −3.25 |
| Claude Opus 5 | −7.60 | −11.31, −3.89 |
| Qwen3.7 Max | −9.34 | −12.93, −5.75 |
| Mistral Large 2512 | −9.68 | −13.27, −6.09 |
| Grok 4.5 | −11.30 | −14.85, −7.76 |
| Llama 4 Maverick | −12.52 | −15.96, −9.07 |
| Gpt 5.6 Terra | −13.35 | −17.10, −9.59 |
| Deepseek V4 Pro | −14.76 | −18.54, −10.98 |

> Interpretation caveat from the paper: the 50-vignette set was **selected using prior model
> scores**, reviewers were a convenience sample, and human ratings are a **reference standard, not
> ground truth**. Read these as evidence of systematic divergence in this sample — not proof that
> models underestimate patient capacity.

---

## The instrument

Thirteen items across four domains, each scored **0–4** (0 = absent, 1 = minimal, 2 = moderate,
3 = high, 4 = very high / fully present) with an explicit **N/A** option:

**Value Awareness (VA)**
- `VA1` Value Clarity — how clearly the subject articulates core values relevant to the decision
- `VA2` Value Stability — how stable and consistent those expressed values are
- `VA3` Framework Awareness — awareness of how their values relate to the decision
- `VA4` Appreciation — appreciation of the clinical team's efforts to provide comfort and care

**Factual Understanding (FU)**
- `FU1` Key Facts Recall — recall of key facts about condition and treatment options
- `FU2` Risk Comprehension — understanding of risks and benefits of each option
- `FU3` Applicability — ability to apply the information to their own situation

**Rational Deliberation (RD)**
- `RD1` Coherence — logical coherence and consistency of reasoning
- `RD2` Trade-off Reasoning — ability to weigh trade-offs between options
- `RD3` Consistency — decision consistent with stated values and understanding

**Intentional Action (IA)**
- `IA1` Intention Strength — clarity of the intended course of action
- `IA2` Planfulness — consideration of practical implementation steps
- `IA3` Follow-through Feasibility — whether the plan is realistic and feasible

**Contextual modifiers**, scored **0–100** and reported separately (never folded into the index):
- `ECI` External Constraint Index — external constraints on autonomy
- `SPI` Support Provided Index — support provided to the subject

### Scoring

Each domain is rescaled to 0–100 as `(sum of items) / (4 × n_items) × 100`. The composite is an
equal-weight mean of the four domains:

```
AIx = 0.25·VA + 0.25·FU + 0.25·RD + 0.25·IA
```

Interpretation bands: **≥ 80** strong evidence of autonomous behavior · **60–79** adequate autonomy,
additional supports may optimize · **< 60** autonomy not fully demonstrated, reassessment
recommended.

Reference implementation: `AutonomyIndexCalculator` in
`code/scoring pipeline/LLM_scoring_cases_open_router.py`.

---

## Repository layout

```
.
├── Autonomy_Index_medRxiv_v30.docx     # manuscript draft
├── data/
│   ├── vignettes/
│   │   ├── published_vignettes.json    # 50 vignettes (id, title, text, source, category, tags)
│   │   └── Survey_Questions.xlsx       # the instrument as presented to raters and models
│   └── scores/
│       ├── human_Sept_7_cleaned_sharing.csv   # de-identified human item-level ratings
│       ├── scores_july_26_llm_values_50.csv   # July 2026 LLM panel, 8 models × 10 runs × 50 cases
│       └── march_vignette_scores.csv          # March 2026 LLM panel, 9 models × 89 cases
├── code/
│   ├── scoring pipeline/
│   │   └── LLM_scoring_cases_open_router.py   # OpenRouter batch scoring + index computation
│   └── analysis/
│       ├── autonomy_statistical.ipynb         # descriptive, divergence, Overton pluralistic
│       ├── autonomy_psychometerics.ipynb      # G-theory, reliability, agreement, bias models
│       └── autonomy_causal.ipynb              # variance decomposition, moderation, per-model gaps
└── outcomes/
    ├── complete_outcomes/     # fig01–fig14, complete_report.txt, summary CSVs
    ├── psychometrics/         # results.json, interpreted_results.txt, 3 figures
    └── causal_outcomes/       # causal_*.csv, figCC1–figCC6, distribution figure
```

### Data files

**`data/vignettes/published_vignettes.json`** — 50 vignettes. Fields: `id`, `title`,
`vignette_text`, `source_type` (48 `literature`, 2 `original`), `source_citation`, `category`,
`tags`. All cases derive from published case reports and public-domain clinical material and
contain no direct or indirect patient identifiers. Three widely taught cases (Quinlan, Tarasoff,
Baby K) are included deliberately as anchors.

**`data/scores/human_Sept_7_cleaned_sharing.csv`** — 191 de-identified rating rows across 51 case
ids (the 50-vignette comparison set plus one additional case); 173 ratings enter the primary
comparison. Columns: `case_id`, `case_title`, the 13 items, `ECI`, `SPI`. Blank item cells are
reviewer N/A responses. Individual reviewer identifiers and registration data are **not** shared —
participants did not consent to release of individual-level records.

**`data/scores/scores_july_26_llm_values_50.csv`** — 4,000 rows from the July 2026 panel
(3,431 `status == "ok"`, 568 `incomplete`, 1 `json_error`). Eight models — Claude Opus 5,
Deepseek V4 Pro, Gemini 3.5 Flash Lite, Gpt 5.6 Terra, Grok 4.5, Llama 4 Maverick,
Mistral Large 2512, Qwen3.7 Max — scored all 50 vignettes across ~10 runs each. Carries raw items,
domain scores, `autonomy_index`, `interpretation`, plus per-call provenance (`model_id`, `arm`,
`provider`, `parse_mode`, `finish_reason`, `completion_tokens`, `cost_usd`, `attempts`,
`latency_s`, `ts`, `error`).

**`data/scores/march_vignette_scores.csv`** — 7,524 rows from the earlier March 2026 panel
(9 models, 89 vignettes) used for the score-tier and inter-model-disagreement selection that
produced the 50-vignette comparison set.

---

## Reproducing the analysis

### Requirements

Python 3.10+ with `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`, `factor_analyzer`,
`tqdm`. The scoring pipeline additionally needs `openai` (used as the OpenRouter client) and
`python-dotenv`. A couple of psychometric steps (Kenward–Roger corrected F-tests) shell out to
`Rscript` and degrade gracefully if R is absent.

```bash
pip install numpy pandas matplotlib scipy statsmodels factor_analyzer tqdm openai python-dotenv
```

### 1. Score vignettes with LLMs

```bash
export OPENROUTER_API_KEY=sk-or-...        # or put it in a .env file
python "code/scoring pipeline/LLM_scoring_cases_open_router.py"
```

The model panel is the `OPENROUTER_MODELS` dict near the top of the script (successors, frozen
anchors, and an optional exploratory arm, each commented). Model ids are validated against the live
OpenRouter catalog before any scoring begins, so a retired or renamed slug fails loudly instead of
silently switching providers. Each call asks for a bare JSON object of the 15 scores; the script
parses, validates ranges (0–4 for items, 0–100 for ECI/SPI), computes the index, and writes a
per-run CSV plus per-vignette radar charts.

### 2. Run the analysis notebooks

| Notebook | Produces |
|---|---|
| `autonomy_statistical.ipynb` | `outcomes/complete_outcomes/` — descriptive comparison, KL/JS divergence, Overton pluralistic coverage, `fig01`–`fig14`, `complete_report.txt` |
| `autonomy_psychometerics.ipynb` | `outcomes/psychometrics/` — N/A rates, G-theory variance components and D-study, ordinal α, within-model self-consistency, Bland–Altman, Holm–Bonferroni item tests, landmark-case checks |
| `autonomy_causal.ipynb` | `outcomes/causal_outcomes/` — variance decomposition, rater-source effect, difficulty and ECI/SPI moderation, per-model forest plot |

**Path note:** the notebooks were run in Colab and hard-code `DATA_DIR = '/content/sample_data'`
with filenames `human_july_14_cleaned.csv`, `scores_july_26_llm_values_50.csv`, and
`scores_march_26_llm_values_50.csv`. To run locally, point `DATA_DIR` at `data/scores/` and map the
human file to `human_Sept_7_cleaned_sharing.csv` and the March file to `march_vignette_scores.csv`.
`OUT_DIR` is derived from `os.getcwd()`; set it to the matching `outcomes/` subdirectory.
The scoring pipeline's `load_questions` / `load_vignettes` defaults also point at absolute paths
from the author's machine — repoint them at `data/vignettes/`.

---

## Selected secondary findings

**Reliability and consistency.** Ordinal α is excellent for both sources (LLM 0.978, human 0.971).
July LLMs show high single-rater generalizability (ρ₁ = 0.837) and high run-to-run self-consistency
(ICC₁ > 0.90 for 6 of 8 models; Mistral Large 2512 highest at 0.998, Deepseek V4 Pro lowest at
0.825).

**Distributional divergence** (Jensen–Shannon distance, human vs LLM): Autonomy Index 0.257
(*small*); Value Awareness 0.286 (*small*); Factual Understanding 0.344 (*moderate*); Rational
Deliberation 0.397 (*moderate*); Intentional Action 0.460 (*large*). At item level most items are
negligible; the most divergent are `IA2` Planfulness, `IA3` Follow-through Feasibility, `FU1` Key
Facts Recall, and `VA4` Appreciation. See `outcomes/complete_outcomes/divergence_summary.csv`.

**Overton pluralistic coverage** — does the model span the *range* of reasonable human views, not
just the mean? Overall vignette-level coverage mean 0.741 (median 0.949); 38 of 50 vignettes reach
coverage ≥ 0.5. Best composite pluralistic alignment: Qwen3.7 Max (0.864) and Gpt 5.6 Terra (0.863);
lowest: Deepseek V4 Pro (0.765).

**Landmark cases.** Among the well-anchored cases (≥ 4 human raters), several show large gaps —
Clinical Trial Consent in Alzheimer's (+22.0), "Too Little, too Late… Almost" (+20.8), Vaccine
Hesitancy (+17.4), "Everything Be Done" (+16.8) — while Baby K and "When a Parent Overrules Their
Child" run slightly the other way.

Full narrative with per-analysis rationale: `outcomes/psychometrics/interpreted_results.txt` and
`outcomes/complete_outcomes/complete_report.txt` (the latter also documents what every figure shows).

---

## Ethics and data availability

Human data collection was reviewed and approved by the **Harvard Longwood Campus IRB, protocol
IRB26-0146**. All participants gave documented electronic informed consent, archived separately
from response data. No identifiable patient data were used. The study was conducted in accordance
with the Declaration of Helsinki and the Belmont Report.

De-identified vignette-level and item-level rating data and the full instrument are also available
at <https://medethics.org/>. Individual reviewer identifiers and registration data are not shared.

## Citation

```bibtex
@article{duttaroy_autonomyindex,
  title   = {Auditing the Human--LLM Autonomy Gap in Clinical Ethics: Development and
             Application of the Autonomy Index Across 50 Clinical Ethics Vignettes},
  author  = {Dutta Roy, Taposh and Brendel, Rebecca Weintraub},
  journal = {medRxiv},
  note    = {Preprint},
  url     = {https://github.com/taposh/autonomyindex}
}
```

**Corresponding author:** Taposh Dutta Roy, Center for Bioethics, Harvard Medical School,
641 Huntington Avenue, Boston, MA 02115, USA — taposh_duttaroy@hms.harvard.edu
ORCID: [0000-0002-8166-3636](https://orcid.org/0000-0002-8166-3636) ·
R. W. Brendel [0009-0005-9300-7029](https://orcid.org/0009-0005-9300-7029)

## Competing interests

T.D.R. is employed as Director of Innovation and Artificial Intelligence at Kaiser Permanente. This
work was conducted in his capacity as a graduate student researcher at the Harvard Medical School
Center for Bioethics, not on behalf of or with resources from that employer.

## License

Code is released under the **MIT License** — see [LICENSE](LICENSE). Vignette and rating data are
shared for research use subject to the IRB and consent terms above; vignettes retain the rights of
their original published sources, cited per-vignette in `published_vignettes.json`.

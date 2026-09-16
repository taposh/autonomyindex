# %%
#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations
"""
──────────────────────────────────────────────────────────────────────────────
Scores clinical vignettes using 13 OpenRouter models (Claude already done).
Saves results to vignette_scores_test_3_6.csv and per-vignette radar PNGs.

Target models (OpenRouter only — Claude excluded):
  GPT-5.4, Gemini 3 Pro Preview, Grok 4.1 Fast, MiniMax M2.5,
  Kimi K2.5, DeepSeek V3.2, Llama 3.3 70B, Llama 4 Scout,
  Qwen3 235B Thinking, Mistral Large 3, Devstral, Step 3.5 Flash,
  Trinity Large Preview

Env vars:
    OPENROUTER_API_KEY   sk-or-...   (only OpenRouter key needed)
"""

# ── Cell 1: Imports & env ──────────────────────────────────────────────────────
import os, re, math, json, getpass, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — remove if running interactively
import matplotlib.pyplot as plt
from tqdm.auto import tqdm     # NEW: progress bar

from dataclasses import dataclass, field

#import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
print("OPENROUTER_API_KEY loaded:", bool(os.getenv("OPENROUTER_API_KEY")))




# %%
# ── Cell 2: Model catalogue ────────────────────────────────────────────────────
#March_panel
# Only the 13 OpenRouter models — Claude already completed
'''
OPENROUTER_MODELS: dict[str, str] = {
    "GPT-5.4":               "openai/gpt-5.4",
    "Gemini 3 Flashlite Preview":  "google/gemini-3.1-flash-lite-preview",
    "Grok 4.1 Fast":         "x-ai/grok-4.1-fast",
    #"MiniMax M2.5":          "minimax/minimax-m2.5-20260211",
    #"Kimi K2.5":             "moonshotai/kimi-k2.5-0127",
    "DeepSeek V3.2":         "deepseek/deepseek-v3.2-20251201",
    "Llama 3.3 70B":         "meta-llama/llama-3.3-70b-instruct",
    "Llama 4 Scout":         "meta-llama/llama-4-scout",
    "Qwen 2.5 72B":      "qwen/qwen-2.5-72b-instruct",
    "Command R+":        "cohere/command-r-plus",
    #"Qwen3 235B Thinking":   "qwen/qwen3-235b-a22b-thinking-2507",
    #"Mistral Large 3":       "mistralai/mistral-large-3",
    #"Devstral":              "mistralai/devstral",
    #"Step 3.5 Flash":        "stepfun/step-3.5-flash",
    #"Trinity Large Preview": "arcee-ai/trinity-large-preview",
}

ALL_MODELS: list[str] = list(OPENROUTER_MODELS.keys())
print(f"Models to run: {len(ALL_MODELS)}")
print(ALL_MODELS)

'''


# %%
# July 2026 panel — Claude run separately via Anthropic API
# Tier 2 = generational successors to the March run
# Anchors = unchanged weights, re-run to establish the noise floor
OPENROUTER_MODELS: dict[str, str] = {
    # --- successors ---
    "GPT-5.6 Terra":         "openai/gpt-5.6-terra",           # verify
    "GPT-5.6 Sol":           "openai/gpt-5.6-sol",             # verify
    "Gemini 3.5 Flash-Lite":  "google/gemini-3.5-flash-lite",  # tier-matched successor
                                                               # to the March model
    #"Gemini 3.1 Pro":        "google/gemini-3.1-pro-preview", # newest Pro, but PREVIEW —
                                                               # tier jump + retirable slug
    "Grok 4.5":              "x-ai/grok-4.5",
    "DeepSeek V4 Pro":       "deepseek/deepseek-v4-pro",       # verify
    "Qwen3.6 27B":           "qwen/qwen3.6-27b",               # NB: dense 27B, not a
                                                               # size-matched pair for
                                                               # the 72B anchor
    "Mistral Large 3":       "mistralai/mistral-large-2512",   # date-stamped slug
    # --- frozen anchors (unchanged from March) ---
    "Gemini 3.1 Flash-Lite":  "google/gemini-3.1-flash-lite-preview",  # the March model,
                                                                       # still available
    "Llama 3.3 70B":         "meta-llama/llama-3.3-70b-instruct",
    "Llama 4 Scout":         "meta-llama/llama-4-scout",
    "Qwen 2.5 72B":          "qwen/qwen-2.5-72b-instruct",
    "Command R+":            "cohere/command-r-plus-08-2024",  # undated alias retired
    # --- optional / exploratory arm ---
    #"GPT-5.6 Luna":          "openai/gpt-5.6-luna",
    #"Gemini 3.6 Flash":      "google/gemini-3.6-flash",
    #"DeepSeek V4 Flash":     "deepseek/deepseek-v4-flash",
    #"GLM-5.2":               "z-ai/glm-5.2",
    #"Kimi K3":               "moonshotai/kimi-k3",            # weights 7/27
    #"Muse Spark 1.1":        "meta-llama/muse-spark-1.1",
}

# Pin the upstream provider per model. OpenRouter routes one model ID across
# several providers with different quantizations and different endpoint support;
# unpinned, that variance lands inside your model-identity term. Critical for the
# frozen anchors, whose entire job is to be identical to the March run.
# Populate from list_models.py / the /endpoints API. Absent = OpenRouter picks.
PROVIDER_PINS: dict[str, str] = {
    "Qwen 2.5 72B": "DeepInfra",     # Novita rejects the chat endpoint outright
}

# Constrained decoding via response_format={"type":"json_object"}.
#
# DEFAULT OFF. It removes most malformed-JSON failures, but it restricts the
# token distribution at sampling time and therefore SUPPRESSES run-to-run
# variance — and only for the subset of models that support it. In a study
# estimating per-vignette variance across repeated runs, that is a confound on
# the dependent variable, and it would not be comparable to the March run.
# Turn on only if March used it too, and only if applied to every model.
USE_JSON_MODE: bool = False

JSON_MODE_OK: set[str] = {
    "GPT-5.6 Terra", "GPT-5.6 Sol", "Grok 4.5", "DeepSeek V4 Pro",
    "Mistral Large 3", "Gemini 3.5 Flash-Lite", "Gemini 3.1 Flash-Lite",
} if USE_JSON_MODE else set()

ALL_MODELS: list[str] = list(OPENROUTER_MODELS.keys())
print(f"Models to run: {len(ALL_MODELS)}")
print(ALL_MODELS)
 

# %%


# ── Cell 3: LLM Registry ───────────────────────────────────────────────────────

@dataclass
class LLMRegistry:
    """Holds initialised API clients."""
    _openrouter_client: OpenAI | None = field(default=None, repr=False)

    def init_openrouter(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            key = getpass.getpass("OpenRouter API key (hidden): ").strip()
        if not key:
            raise ValueError("OpenRouter API key is required.")
        self._openrouter_client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://autonomy-index-research",
                "X-Title":      "Autonomy Index Scorer",
            },
        )
        print("✓ OpenRouter client ready.")

    @property
    def openrouter(self) -> OpenAI:
        if self._openrouter_client is None:
            raise RuntimeError("Call init_openrouter() first.")
        return self._openrouter_client


def build_registry() -> LLMRegistry:
    """Initialise OpenRouter client from env var or interactive prompt."""
    registry = LLMRegistry()
    registry.init_openrouter()
    return registry


def call_llm(
    registry:   LLMRegistry,
    model_name: str,
    prompt:     str,
    max_tokens: int   = 4000,
    retries:    int   = 3,
    retry_wait: float = 5.0,
    diag:       dict | None = None,
) -> str:
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {ALL_MODELS}")

    model_id = OPENROUTER_MODELS[model_name]

    extra: dict = {}
    pin = PROVIDER_PINS.get(model_name)
    if pin:
        # allow_fallbacks=False: better a clean 429 we back off from than a
        # silent switch to a different provider serving different weights.
        extra["provider"] = {"order": [pin], "allow_fallbacks": False}

    kwargs: dict = {}
    if model_name in JSON_MODE_OK:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(1, retries + 1):
        try:
            resp = registry.openrouter.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra or None,
                **kwargs,
            )

            if not getattr(resp, "choices", None):
                err = getattr(resp, "error", None) or "no choices returned"
                raise RuntimeError(f"{model_name}: {err}")

            choice = resp.choices[0]
            msg    = choice.message
            fr     = getattr(choice, "finish_reason", None)

            # Reasoning models return content=None when the whole token budget
            # went to the reasoning trace. Fall back to the reasoning field.
            content = getattr(msg, "content", None) or ""
            if not content.strip():
                content = getattr(msg, "reasoning", None) or ""

            if not content.strip():
                usage = getattr(resp, "usage", None)
                ct    = getattr(usage, "completion_tokens", "?") if usage else "?"
                if fr == "tool_calls":
                    # Provider chat template misfired; stochastic, so retryable.
                    raise RuntimeError(
                        f"{model_name}: empty content, finish_reason=tool_calls "
                        f"(provider template emitted a tool call; retrying)"
                    )
                raise RuntimeError(
                    f"{model_name}: empty response, not retried "
                    f"(finish_reason={fr}, completion_tokens={ct}, "
                    f"max_tokens={max_tokens})"
                )

            if diag is not None:
                usage = getattr(resp, "usage", None)
                diag.update(
                    provider        = getattr(resp, "provider", None),
                    finish_reason   = fr,
                    prompt_tokens   = getattr(usage, "prompt_tokens", None) if usage else None,
                    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None,
                    attempts        = attempt,
                    used_reasoning_field = not (getattr(msg, "content", None) or "").strip(),
                )
            return content.strip()

        except (ValueError, KeyError):
            raise
        except Exception as exc:
            txt = str(exc)

            # A 429 anywhere in the chain is transient — including a 400 that is
            # really "primary provider was rate-limited, fallback rejected the
            # request". The old heuristic saw "400" and refused to retry, which
            # turned a temporary rate limit into 10 permanently lost runs.
            transient = ("429" in txt
                         or "rate-limit" in txt.lower()
                         or "rate limit" in txt.lower()
                         or "temporarily" in txt.lower()
                         or "timeout" in txt.lower()
                         or "tool_calls" in txt
                         or getattr(exc, "status_code", None) in (429, 500, 502, 503, 529))

            permanent = (not transient) and (
                "not retried" in txt
                or "not a valid model ID" in txt
                or "does not support endpoint" in txt
                or getattr(exc, "status_code", None) == 400
                or txt.startswith("Error code: 400")
            )

            if permanent or attempt == retries:
                raise

            # Exponential backoff with a longer floor for rate limits.
            wait = (retry_wait * 3 if "429" in txt or "rate" in txt.lower()
                    else retry_wait) * (2 ** (attempt - 1))
            print(f"      \u26a0  {model_name} attempt {attempt} failed "
                  f"({txt[:110]}) \u2014 retrying in {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError("Unreachable")


def preflight_check(registry: LLMRegistry) -> None:
    """
    NEW: validate every model ID against the live catalogue BEFORE scoring.
    Two seconds here would have saved this run. Aborts with the bad IDs listed.
    """
    print("Preflight: validating model IDs against OpenRouter catalogue...")
    catalog = {m.id for m in registry.openrouter.models.list().data}
    bad = {name: mid for name, mid in OPENROUTER_MODELS.items()
           if name in ALL_MODELS and mid not in catalog}
    if bad:
        print(f"\n  {len(bad)} INVALID MODEL ID(S):")
        for name, mid in bad.items():
            author = mid.split("/")[0]
            near = sorted(c for c in catalog if c.startswith(author + "/"))
            print(f"    {name:22} '{mid}'")
            for c in near:                     # full list — truncating hid the answer
                print(f"        {c}")
        raise SystemExit("\nFix OPENROUTER_MODELS and re-run. Nothing was scored.")
    print(f"  All {len(ALL_MODELS)} model IDs valid.\n")

# %%
# ── Cell 4: Load data ──────────────────────────────────────────────────────────

def load_questions(path: str = "/Users/taposhduttaroy/workspace/code/baia/data/survey_questions_for_code.csv") -> pd.DataFrame:
    df = pd.read_csv(path)[["ID", "Label", "Description"]].copy()
    df["Label"]       = df["Label"].str.strip()
    df["Description"] = df["Description"].str.strip()
    print(f"✓ Loaded {len(df)} survey questions.")
    return df


def load_vignettes(path: str = "/Users/taposhduttaroy/workspace/code/baia/data/cases2.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ["id", "title", "vignette_text"]:
        df[col] = df[col].str.strip()
    print(f"✓ Loaded {len(df)} vignettes.")
    return df



# %%
# ── Cell 5: Scoring & Autonomy Index ──────────────────────────────────────────

def scoring_func(question_id: str, score) -> float:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return float("nan")
    score = float(score)
    low, high = (0, 100) if question_id in ("ECI", "SPI") else (0, 4)
    if not (low <= score <= high):
        raise ValueError(
            f"Score {score!r} out of range for '{question_id}'. Expected {low}–{high}."
        )
    return score


class AutonomyIndexCalculator:
    def __init__(self):
        self.default_weights = {"VA": 0.25, "FU": 0.25, "RD": 0.25, "IA": 0.25}

    def _rescale(self, raw_scores):
        return (sum(raw_scores) / (4 * len(raw_scores))) * 100

    def get_va_score(self, clarity, stability, awareness, appreciation):
        return self._rescale([clarity, stability, awareness, appreciation])

    def get_fu_score(self, recall, risk_comp, applicability):
        return self._rescale([recall, risk_comp, applicability])

    def get_rd_score(self, coherence, trade_off, consistency):
        return self._rescale([coherence, trade_off, consistency])

    def get_ia_score(self, strength, planfulness, feasibility):
        return self._rescale([strength, planfulness, feasibility])

    def calculate_composite_ai(self, va, fu, rd, ia, weights=None):
        w = weights if weights else self.default_weights
        return (w["VA"] * va) + (w["FU"] * fu) + (w["RD"] * rd) + (w["IA"] * ia)

    def interpret_ai(self, score):
        # NEW: a NaN mean (all runs failed) fell through both comparisons and
        # was labelled "Autonomy not fully demonstrated" on the radar title.
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return "Not scored"
        if score >= 80: return "Strong evidence of autonomous behavior"
        if score >= 60: return "Adequate autonomy; additional supports may optimize"
        return "Autonomy not fully demonstrated; reassessment recommended"


def safe(score: float, fallback: float = 0.0) -> float:
    return fallback if (score is None or math.isnan(score)) else score


def compute_autonomy_index(scores: dict,
                            calc: AutonomyIndexCalculator,
                            weights: dict = None) -> dict:
    va = calc.get_va_score(
        clarity=safe(scores["VA1"]), stability=safe(scores["VA2"]),
        awareness=safe(scores["VA3"]), appreciation=safe(scores["VA4"]),
    )
    fu = calc.get_fu_score(
        recall=safe(scores["FU1"]), risk_comp=safe(scores["FU2"]),
        applicability=safe(scores["FU3"]),
    )
    rd = calc.get_rd_score(
        coherence=safe(scores["RD1"]), trade_off=safe(scores["RD2"]),
        consistency=safe(scores["RD3"]),
    )
    ia = calc.get_ia_score(
        strength=safe(scores["IA1"]), planfulness=safe(scores["IA2"]),
        feasibility=safe(scores["IA3"]),
    )
    ai = calc.calculate_composite_ai(va, fu, rd, ia, weights=weights)
    return {
        "va_score":       round(va, 2),
        "fu_score":       round(fu, 2),
        "rd_score":       round(rd, 2),
        "ia_score":       round(ia, 2),
        "autonomy_index": round(ai, 2),
        "interpretation": calc.interpret_ai(ai),
    }


# ── Cell 6: Prompt & parsing ───────────────────────────────────────────────────

def build_prompt(title: str, text: str, questions: pd.DataFrame) -> str:
    q_lines = "\n".join(
        f'  "{r["ID"]}": "{r["Label"]} – {r["Description"]}"'
        for _, r in questions.iterrows()
    )
    return f"""You are a clinical bioethics rater. Read the vignette and score each question.

=== VIGNETTE: {title} ===
{text}

=== SURVEY QUESTIONS ===
{q_lines}

=== SCORING SCALES ===
  VA1–VA4, FU1–FU3, RD1–RD3, IA1–IA3 → integer 0–4
    (0=absent, 1=minimal, 2=moderate, 3=high, 4=very high/fully present; null=N/A)
  ECI, SPI → integer 0–100
    (0=none, 50=moderate, 100=extreme; null=N/A)

Return ONLY a valid JSON object — no preamble, no explanation, no markdown fences.

Example:
{{
  "VA1": 3, "VA2": 2, "VA3": 3, "VA4": 2,
  "FU1": 3, "FU2": 3, "FU3": 2,
  "RD1": 3, "RD2": 2, "RD3": 3,
  "IA1": 2, "IA2": 1, "IA3": 1,
  "ECI": 30, "SPI": 60
}}
"""


def _extract_json_block(text: str) -> str:
    """Return the first balanced {...} block, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No '{' in response", text, 0)
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:            esc = False
            elif c == "\\":    esc = True
            elif c == '"':     in_str = False
            continue
        if c == '"':           in_str = True
        elif c == "{":         depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # Unterminated (truncated mid-object): salvage the complete key:value pairs.
    return text[start:]


def _repair_json(block: str) -> str:
    """Repair the malformations weaker models actually produce."""
    b = block
    b = re.sub(r",\s*([}\]])", r"\1", b)               # trailing commas
    b = re.sub(r"'([A-Za-z_][A-Za-z0-9_]*)'\s*:", r'"\1":', b)   # single-quoted keys
    b = re.sub(r"(?<![\"\w])([A-Za-z_][A-Za-z0-9_]*)\s*:(?=\s*[\d\-\"n])",
               r'"\1":', b)                           # bare keys
    b = re.sub(r":\s*(null|NULL|None|N/A|NA)\s*([,}])", r": null\2", b)
    if b.count("{") > b.count("}"):                   # truncated object
        b = b.rstrip().rstrip(",")
        b = re.sub(r',\s*"[^"]*"\s*:\s*[^,}]*$', "", b)   # drop partial last pair
        b = re.sub(r',\s*"[^"]*$', "", b)                 # drop unterminated string
        b += "}" * (b.count("{") - b.count("}"))
    return b


def parse_and_validate(raw_json: str, questions: pd.DataFrame,
                       diag: dict | None = None) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw_json).strip()

    raw, mode = None, None
    for label, candidate in (("strict",   cleaned),
                             ("extracted", _extract_json_block(cleaned)),
                             ("repaired",  _repair_json(_extract_json_block(cleaned)))):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            raw, mode = parsed, label
            break
    if diag is not None:
        diag["parse_mode"] = mode
    if raw is None:
        raise json.JSONDecodeError(
            f"Unparseable after repair; first 160 chars: {cleaned[:160]!r}", cleaned, 0)

    validated = {}
    for _, row in questions.iterrows():
        qid = row["ID"]
        try:
            validated[qid] = scoring_func(qid, raw.get(qid))
        except (ValueError, TypeError) as e:
            print(f"    ⚠  {e} – storing NaN")
            validated[qid] = float("nan")
    return validated


# ── Cell 7: Radar plots ────────────────────────────────────────────────────────

_DOMAIN_LABELS_SHORT = ["VA", "FU", "RD", "IA"]
_DOMAIN_LABELS_LONG  = ["Value\nAwareness", "Factual\nUnderstanding",
                         "Rational\nDeliberation", "Intentional\nAction"]
_DOMAIN_COLS         = ["va_score", "fu_score", "rd_score", "ia_score"]
_DOMAIN_COLORS       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
_N_AXES              = 4
_ANGLES              = np.linspace(0, 2 * np.pi, _N_AXES, endpoint=False).tolist() + [0]


def _make_radar_axes(ax, labels=_DOMAIN_LABELS_SHORT, label_size=9):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 100)
    ax.set_rticks([25, 50, 75, 100])
    ax.set_rlabel_position(30)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_thetagrids(np.degrees(_ANGLES[:-1]), labels, fontsize=label_size)


def plot_vignette_radar_multi_run(title: str,
                                   vignette_id: str,
                                   model_results: list[dict],   # ← fixed param name
                                   save_path: str | None = None,
                                   show: bool = False) -> None:
    """
    Overlay one polygon per model on a single radar chart.
    Each polygon = that model's mean score across n_runs.
    Bold dashed black line = cross-model mean.
    """
    if save_path is None:
        save_path = f"radar_multi_{str(vignette_id)[:8]}.png"

    n_models = len(model_results)
    palette  = plt.cm.tab20(np.linspace(0, 1, n_models))

    all_vals  = np.array([[r[c] for c in _DOMAIN_COLS] for r in model_results], dtype=float)
    mean_vals = all_vals.mean(axis=0)
    std_vals  = all_vals.std(axis=0)

    ai_scores = [r["autonomy_index"] for r in model_results
                 if not math.isnan(r["autonomy_index"])]
    ai_mean = np.mean(ai_scores) if ai_scores else float("nan")
    ai_std  = np.std(ai_scores)  if ai_scores else float("nan")

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    _make_radar_axes(ax, labels=_DOMAIN_LABELS_LONG, label_size=9)

    for result, color in zip(model_results, palette):
        vals   = [result[c] for c in _DOMAIN_COLS]
        v_plot = vals + [vals[0]]
        label  = result.get("model_name", "?")
        ax.plot(_ANGLES, v_plot, color=color, linewidth=1.2, alpha=0.6, label=label)
        ax.fill(_ANGLES, v_plot, color=color, alpha=0.06)

    mv_plot = mean_vals.tolist() + [mean_vals[0]]
    ax.plot(_ANGLES, mv_plot, color="black", linewidth=2.5,
            linestyle="--", label=f"Mean (n={n_models})", zorder=5)

    for angle, mu, sd in zip(_ANGLES[:-1], mean_vals, std_vals):
        ax.text(angle, min(mu + 9, 98), f"{mu:.0f}±{sd:.0f}",
                ha="center", va="center", fontsize=7,
                color="black", fontweight="bold")

    ax.legend(loc="upper right", bbox_to_anchor=(1.55, 1.15),
              fontsize=7, framealpha=0.7, ncol=1)

    short = str(title)[:52] + ("…" if len(str(title)) > 52 else "")
    ax.set_title(f"{short}\nAI = {ai_mean:.1f} ± {ai_std:.1f}  |  {n_models} models",
                 fontsize=9, pad=22)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()




# %%
# ── Cell 7b: Smoke test — one call per model before committing to the full run ─

def smoke_test(registry,
               questions: pd.DataFrame,
               vignettes: pd.DataFrame,
               models: list[str] | None = None,
               vignette_idx: int = 0,
               n_runs_planned: int = 10,
               require_all: bool = True) -> list[str]:
    """
    One real scoring call per model against a real vignette, before spending
    the full run. Verifies end to end: the ID resolves, the provider serves it,
    content comes back, the JSON parses, and all 15 items score.

    Preflight checks that a model ID EXISTS. This checks that it WORKS.

    Returns the list of models that passed. With require_all=True (default),
    raises SystemExit if any model fails, so nothing is spent on a broken panel.
    """
    models = models or ALL_MODELS
    vrow   = vignettes.iloc[vignette_idx]
    prompt = build_prompt(vrow["title"], vrow["vignette_text"], questions)
    calc   = AutonomyIndexCalculator()

    print("=" * 100)
    print(f"SMOKE TEST — 1 call x {len(models)} models on vignette "
          f"{vignette_idx + 1}: {str(vrow['title'])[:56]}")
    print("=" * 100)

    rows, passed, failed = [], [], []

    for name in tqdm(models, desc="Smoke test", unit="model"):
        diag: dict = {}
        t0 = time.time()
        try:
            raw = call_llm(registry, name, prompt, diag=diag)
            scores = parse_and_validate(raw, questions, diag=diag)
            ai     = compute_autonomy_index(scores, calc)

            item_ids = [q for q in questions["ID"]]
            n_ok = sum(1 for q in item_ids
                       if not (isinstance(scores.get(q), float)
                               and math.isnan(scores[q])))
            ok = n_ok == len(item_ids)
            (passed if ok else failed).append(name)
            rows.append({
                "model":    name,
                "status":   "PASS" if ok else f"PARTIAL {n_ok}/{len(item_ids)}",
                "secs":     round(time.time() - t0, 1),
                "provider": diag.get("provider") or "-",
                "parse":    diag.get("parse_mode") or "-",
                "finish":   diag.get("finish_reason") or "-",
                "out_tok":  diag.get("completion_tokens") or "-",
                "AI":       ai["autonomy_index"],
                "note":     "" if ok else "some items missing from response",
            })
        except Exception as e:
            failed.append(name)
            rows.append({
                "model": name, "status": "FAIL",
                "secs": round(time.time() - t0, 1),
                "provider": diag.get("provider") or "-",
                "parse": diag.get("parse_mode") or "-",
                "finish": diag.get("finish_reason") or "-",
                "out_tok": diag.get("completion_tokens") or "-",
                "AI": float("nan"),
                "note": str(e)[:78],
            })

    df = pd.DataFrame(rows)
    print()
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(df.to_string(index=False))

    # ── things worth knowing before a 30-hour run ────────────────────────────
    repaired = df[df["parse"].isin(["extracted", "repaired"])]["model"].tolist()
    if repaired:
        print(f"\n  NOTE: needed JSON repair — {', '.join(repaired)}")
        print("        Parsed fine, but this is degraded instruction-following.")

    unpinned = [r["model"] for _, r in df.iterrows()
                if r["provider"] not in ("-", None)
                and r["model"] not in PROVIDER_PINS]
    if unpinned:
        print(f"\n  NOTE: no PROVIDER_PINS entry for {len(unpinned)} model(s). "
              f"Observed providers:")
        for _, r in df.iterrows():
            if r["model"] in unpinned:
                print(f"        {r['model']:24} -> {r['provider']}")
        print("        These can silently switch provider mid-run. Pin the anchors.")

    good = df[df["status"] == "PASS"]
    if not good.empty:
        est = (good["secs"].sum() / len(good)) * len(models) * len(vignettes) * n_runs_planned
        print(f"\n  Projected full run: {len(vignettes)} vignettes x {len(models)} models "
              f"x {n_runs_planned} runs = {len(vignettes)*len(models)*n_runs_planned:,} calls"
              f"  ~{est/3600:.1f} h sequential")

    print(f"\n  {len(passed)} passed / {len(failed)} failed")
    if failed:
        print(f"  FAILING: {', '.join(failed)}")
        if require_all:
            raise SystemExit(
                "\nSmoke test failed. Fix the models above, or call "
                "main(require_all_smoke=False) to run only the passing ones.\n"
                "Nothing was spent on the full run."
            )
        print("  Continuing with the passing models only.")
    print("=" * 100 + "\n")
    return passed


# ── Cell 8: Main pipeline ──────────────────────────────────────────────────────

from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

CHECKPOINT_PATH = "scores_checkpoint.jsonl"


def _load_checkpoint(path: str):
    """
    Read completed calls back in. Returns (records, done_keys).

    NEW: only status=="ok" counts as done. Previously ANY recorded row —
    including api_error — was treated as complete, so a bad model ID got
    baked into the checkpoint and could never be retried.
    """
    records, done, failed = [], set(), 0
    if not os.path.exists(path):
        return records, done
    bad = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1          # partial last line from a hard kill — skip it
                continue
            records.append(rec)
            if rec.get("status") == "ok":
                done.add((str(rec["vignette_id"]), rec["model"], int(rec["run"])))
            else:
                failed += 1
    # drop failed rows from the in-memory set so they don't pollute the CSV;
    # they will be re-attempted and a fresh row written
    records = [r for r in records if r.get("status") == "ok"]
    print(f"Resuming: {len(done):,} successful calls already completed"
          + (f", {failed:,} previous failures will be retried" if failed else "")
          + (f" ({bad} truncated line(s) skipped)" if bad else ""))
    return records, done


def _score_one(task, registry, questions, prompts, calc):
    """Score a single (vignette, model, run). Runs inside a worker thread."""
    vi, vid, title, model_name, run = task
    diag: dict = {}
    t0 = time.time()
    try:
        raw       = call_llm(registry, model_name, prompts[vi], diag=diag)
        scores    = parse_and_validate(raw, questions, diag=diag)
        ai_result = compute_autonomy_index(scores, calc)
        status    = "ok"
        err       = ""
    except json.JSONDecodeError as e:
        scores    = {qid: float("nan") for qid in questions["ID"]}
        ai_result = {k: np.nan for k in ["va_score", "fu_score", "rd_score",
                                         "ia_score", "autonomy_index"]}
        ai_result["interpretation"] = "error"
        status, err = "json_error", str(e)[:200]
    except Exception as e:
        scores    = {qid: float("nan") for qid in questions["ID"]}
        ai_result = {k: np.nan for k in ["va_score", "fu_score", "rd_score",
                                         "ia_score", "autonomy_index"]}
        ai_result["interpretation"] = "api_error"
        status, err = "api_error", str(e)[:200]

    record = {
        "vignette_id": vid, "title": title, "model": model_name, "run": run,
        "status": status, "error": err,
        "latency_s": round(time.time() - t0, 2),
        "provider": diag.get("provider"),
        "parse_mode": diag.get("parse_mode"),
        "finish_reason": diag.get("finish_reason"),
        "completion_tokens": diag.get("completion_tokens"),
        "attempts": diag.get("attempts"),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    record.update(scores)
    record.update(ai_result)
    return record


def main(n_runs: int = 10,
         checkpoint: str = CHECKPOINT_PATH,
         resume: bool = True,
         verbose: bool = False,
         max_vignettes: int | None = None,
         smoke: bool = True,
         require_all_smoke: bool = True,
         workers: int = 8,
         runs_by_model: dict[str, int] | None = None):
    """
    Score every vignette with every model.

    Speed
    -----
    The work is I/O-bound — nearly all wall-clock is spent waiting on HTTP
    responses, with the CPU idle. `workers` runs that many calls concurrently,
    which is close to a linear speed-up. workers=1 restores sequential order.
    Keep it at or below ~8: OpenRouter rate-limits per account, and a 429 storm
    produces missingness that correlates with whichever models were in flight.

    `runs_by_model` overrides n_runs per model, e.g.
        runs_by_model={"Qwen 2.5 72B": 10, "GPT-5.6 Terra": 10}
    with n_runs=3 elsewhere — 10 repetitions only where you need a within-model
    variance estimate, 3 where you only need the mean.

    Restart
    -------
    Every completed call is appended to `checkpoint` (JSONL) and flushed.
    Only status=="ok" counts as done, so failures are retried on the next run.
    Re-run the same call after any crash and it continues from where it stopped.
    """
    registry  = build_registry()
    preflight_check(registry)          # stage 1: do the model IDs exist?
    questions = load_questions()
    vignettes = load_vignettes().reset_index(drop=True)
    calc      = AutonomyIndexCalculator()

    if max_vignettes:
        vignettes = vignettes.head(max_vignettes)

    # stage 2: do they actually work end to end?
    models = ALL_MODELS
    if smoke:
        models = smoke_test(registry, questions, vignettes,
                            n_runs_planned=n_runs,
                            require_all=require_all_smoke)
        if not models:
            raise SystemExit("No models passed the smoke test. Nothing was run.")

    os.makedirs("radars", exist_ok=True)
    runs_for = lambda m: (runs_by_model or {}).get(m, n_runs)

    # ── resume ────────────────────────────────────────────────────────────────
    all_records, done = ([], set())
    if resume:
        all_records, done = _load_checkpoint(checkpoint)

    prompts = {i: build_prompt(vignettes.iloc[i]["title"],
                               vignettes.iloc[i]["vignette_text"], questions)
               for i in range(len(vignettes))}

    tasks = [(i, str(vignettes.iloc[i]["id"]), vignettes.iloc[i]["title"], m, r)
             for i in range(len(vignettes))
             for m in models
             for r in range(1, runs_for(m) + 1)]
    todo  = [t for t in tasks if (t[1], t[3], t[4]) not in done]

    total = len(tasks)
    print(f"\nPipeline: {len(vignettes)} vignettes x {len(models)} models "
          f"= {total:,} total calls  ({len(todo):,} remaining)")
    print(f"Concurrency: {workers} worker(s)\n")
    print(f"Models: {models}\n")

    lock    = threading.Lock()
    ckpt    = open(checkpoint, "a" if resume else "w")
    bar     = tqdm(total=len(todo), desc="Scoring", unit="call", smoothing=0.05)
    counts  = {"ok": 0, "err": 0}
    t_start = time.time()

    def _finish(rec):
        with lock:
            all_records.append(rec)
            ckpt.write(json.dumps(rec, default=str) + "\n")
            ckpt.flush()
            os.fsync(ckpt.fileno())
            if rec["status"] == "ok":
                counts["ok"] += 1
            else:
                counts["err"] += 1
                tqdm.write(f"  x {rec['model']} {rec['vignette_id']} "
                           f"run {rec['run']}: {rec['status']}: {rec['error'][:110]}")
            bar.set_postfix_str(f"ok={counts['ok']} err={counts['err']}")
            bar.update(1)

    try:
        if workers <= 1:
            for t in todo:
                _finish(_score_one(t, registry, questions, prompts, calc))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_score_one, t, registry, questions, prompts, calc)
                           for t in todo]
                for fut in as_completed(futures):
                    _finish(fut.result())

    except KeyboardInterrupt:
        tqdm.write("\nInterrupted. Progress is saved — re-run main() to continue.")

    finally:
        ckpt.close()
        bar.close()

        df_results = pd.DataFrame(all_records)
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = f"vignette_scores_july_{stamp}.csv"
        df_results.to_csv(out_csv, index=False)

        # ── radars, built from every record incl. resumed rows ───────────────
        if not df_results.empty:
            domain_keys = ["va_score", "fu_score", "rd_score", "ia_score",
                           "autonomy_index"]
            for vid, grp in df_results.groupby("vignette_id"):
                means = []
                for mname, mgrp in grp.groupby("model"):
                    d = {k: float(pd.to_numeric(mgrp[k], errors="coerce").mean())
                         for k in domain_keys}
                    if math.isnan(d["autonomy_index"]):
                        continue
                    d["model_name"]     = mname
                    d["interpretation"] = calc.interpret_ai(d["autonomy_index"])
                    means.append(d)
                if means:
                    plot_vignette_radar_multi_run(
                        title=str(grp["title"].iloc[0]), vignette_id=vid,
                        model_results=means,
                        save_path=os.path.join("radars", f"radar_multi_{str(vid)[:8]}.png"))
            print(f"  Radars -> radars/")

        elapsed = time.time() - t_start
        rate    = (counts["ok"] + counts["err"]) / elapsed if elapsed else 0
        print(f"\nSaved {len(df_results):,} rows -> '{out_csv}'")
        print(f"  {elapsed/60:.1f} min this session, {rate:.2f} calls/s "
              f"({rate*3600:,.0f}/h)")
        if not df_results.empty and "status" in df_results:
            print("  " + "  ".join(f"{k}={v}" for k, v in
                                   df_results["status"].value_counts().items()))
            print("\nMissing rate by model (watch for systematic failure):")
            print(df_results.groupby("model")["autonomy_index"]
                  .apply(lambda s: pd.to_numeric(s, errors="coerce").isna().mean().round(3))
                  .sort_values(ascending=False))
            # Per-vignette variance is the estimand for this run, so surface it.
            ok = df_results[df_results["autonomy_index"].notna()]
            if not ok.empty and ok.groupby(["vignette_id", "model"]).size().max() > 1:
                v = (ok.groupby(["vignette_id", "model"])["autonomy_index"]
                       .agg(["count", "mean", "std"]))
                v = v[v["count"] > 1]
                if not v.empty:
                    print("\nWithin-cell SD of Autonomy Index (mean across vignettes):")
                    print(v.groupby("model")["std"].mean().round(2)
                            .sort_values(ascending=False))
                    thin = v[v["count"] < 10]
                    if not thin.empty:
                        print(f"  ({len(thin)} vignette x model cells have <10 runs "
                              f"— resume to fill them before analysing variance)")

            if "provider" in df_results:
                multi = (df_results.groupby("model")["provider"]
                         .nunique().pipe(lambda s: s[s > 1]))
                if not multi.empty:
                    print("\n  WARNING: these models were served by >1 provider "
                          "during the run — pin them in PROVIDER_PINS:")
                    for m, n in multi.items():
                        seen = df_results[df_results['model'] == m]['provider'].dropna().unique()
                        print(f"    {m:24} {n} providers: {', '.join(map(str, seen))}")
        print(f"\nCheckpoint: {checkpoint} ({len(done) + counts['ok']:,} ok)")

    return df_results


# ── Cell 9: Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Runs preflight (IDs valid?) then a smoke test (1 call per model, does it
    # actually work?) before spending anything on the full run.
    #
    #   main()                            -> smoke test, abort if any model fails
    #   main(require_all_smoke=False)     -> run only the models that passed
    #   main(smoke=False)                 -> skip it (e.g. resuming a healthy run)
    #
    # Re-run this exact line after any crash / Ctrl-C — it picks up where it
    # stopped. To start over, delete scores_checkpoint.jsonl (or resume=False).
    df_results = main(n_runs=10)

# %%




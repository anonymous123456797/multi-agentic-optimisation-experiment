import json
import os
import time
import re
import warnings
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from radon.complexity import cc_visit
from google import genai
from google.genai import types

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------- CONFIG ----------
INPUT_PATH  = "../MBPP/mbpp_formatted.jsonl"
OUTPUT_PATH = "mbpp_with_difficulty.jsonl"
CHECKPOINT_PATH = "llm_scores_checkpoint.json"
LOG_PATH    = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

MODEL  = "gemini-2.5-flash"
N_RUNS = 2  # Set > 1 to enable fractional bump logic

# Gemini 2.5 Flash pricing (per 1M tokens)
COST_PER_M_INPUT  = 0.30
COST_PER_M_OUTPUT = 2.50

# ---------- ENSEMBLE WEIGHTS ----------
# Must sum to 1.0. LLM carries most weight as it captures semantic difficulty.
# Static metrics act as a tiebreaker / grounding signal.
W_LLM        = 0.70
W_COMPLEXITY = 0.20
W_BRANCHES   = 0.10

assert abs(W_LLM + W_COMPLEXITY + W_BRANCHES - 1.0) < 1e-9, \
    "Ensemble weights must sum to 1.0"

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
log.info(f"Run started. Model: {MODEL}, N_RUNS: {N_RUNS}")
log.info(f"Pricing: ${COST_PER_M_INPUT}/1M input tokens | ${COST_PER_M_OUTPUT}/1M output tokens")
log.info(f"Ensemble weights — LLM: {W_LLM}, Complexity: {W_COMPLEXITY}, Branches: {W_BRANCHES}")

# ---------- INIT GEMINI ----------
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

# ---------- LOAD DATA ----------
data = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                data.append(obj)
        except Exception as e:
            log.warning(f"Bad JSON line {i}: {e}")

if len(data) == 0:
    raise ValueError("No valid data loaded.")

df = pd.DataFrame(data)
log.info(f"Loaded {len(df)} problems from {INPUT_PATH}")

# ---------- VALIDATION ----------
for col in ["code", "text"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: '{col}'")

df["code"] = df["code"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)

# ---------- STRUCTURAL FEATURES ----------
def get_complexity(code: str) -> int:
    """Cyclomatic complexity via radon. Returns 0 on parse failure."""
    try:
        return sum(b.complexity for b in cc_visit(code)) if code else 0
    except Exception:
        return 0

def count_branches(code: str) -> int:
    """Count branching keywords as a lightweight proxy for control-flow depth."""
    keys = ["if ", "for ", "while ", "elif ", "except", "case "]
    return sum(code.count(k) for k in keys)

df["complexity"] = df["code"].apply(get_complexity)
df["branches"]   = df["code"].apply(count_branches)

# ---------- NORMALIZE STATIC FEATURES TO [1, 3] ----------
def normalize_to_scale(series: pd.Series, low: float = 1.0, high: float = 3.0) -> pd.Series:
    """
    Min-max normalize a series to [low, high].
    If the series is constant (min == max), return the midpoint for all rows.
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        mid = low + (high - low) / 2
        log.warning(
            f"Column '{series.name}' is constant (all values = {mn}). "
            f"Setting normalized value to midpoint {mid:.2f} for all rows."
        )
        return pd.Series([mid] * len(series), index=series.index)
    return low + (series - mn) / (mx - mn) * (high - low)

df["complexity_norm"] = normalize_to_scale(df["complexity"].rename("complexity"))
df["branches_norm"]   = normalize_to_scale(df["branches"].rename("branches"))

log.info(
    f"Static features — complexity: mean={df['complexity'].mean():.2f}, "
    f"max={df['complexity'].max()} | branches: mean={df['branches'].mean():.2f}, "
    f"max={df['branches'].max()}"
)

# ---------- LLM HELPERS ----------
PROMPT_TEMPLATE = """You are calibrating difficulty labels for a Python programming benchmark. 
                    Rate this problem relative to the full MBPP dataset.

                    1 = easy (simple step, basic loop or condition)
                    2 = medium (multiple steps, some logic or data structures)
                    3 = hard (requires thinking, non-obvious approach, combining multiple concepts, or easy to get subtly wrong)

                    Reply with a single digit only (1, 2, or 3).

                    Problem: {problem}

                    Solution:
                    {code}"""

def parse_score(text: str):
    """Extract the first digit in [1, 3] from model output. Returns None if not found."""
    if not text:
        return None
    match = re.search(r"[1-3]", text)
    return int(match.group()) if match else None

# ---------- GLOBAL COST / CALL TRACKING ----------
total_input_tokens  = 0
total_output_tokens = 0
total_calls         = 0
failed_calls        = 0

def llm_score(problem: str, code: str, idx: str, retries: int = 3) -> int | None:
    """
    Call the LLM once and return a parsed integer score (1–3), or None on failure.
    Handles retries with exponential back-off and accumulates token / cost stats.
    """
    global total_input_tokens, total_output_tokens, total_calls, failed_calls

    prompt  = PROMPT_TEMPLATE.format(problem=problem, code=code)
    t_start = time.time()

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            latency = time.time() - t_start
            total_calls += 1

            raw_text = response.text.strip() if hasattr(response, "text") else ""
            score    = parse_score(raw_text)

            usage   = getattr(response, "usage_metadata", None)
            in_tok  = getattr(usage, "prompt_token_count",     0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0
            total_input_tokens  += in_tok
            total_output_tokens += out_tok

            call_cost = (
                (in_tok  / 1_000_000 * COST_PER_M_INPUT) +
                (out_tok / 1_000_000 * COST_PER_M_OUTPUT)
            )

            log.info(
                f"[idx={idx}] score={score} raw='{raw_text}' "
                f"in_tok={in_tok} out_tok={out_tok} "
                f"cost=${call_cost:.6f} latency={latency:.2f}s"
            )

            if score is not None:
                return score

            log.warning(f"[idx={idx}] Could not parse score from: '{raw_text}'")

        except Exception as e:
            failed_calls += 1
            log.warning(f"[idx={idx}] LLM call failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)

    log.warning(f"[idx={idx}] All retries failed — no score returned.")
    return None  # caller handles the fallback, not here

# ---------- FRACTIONAL BUMP ----------
def apply_fractional_bump(score: float) -> int:
    """
    Floor the score, then bump up one level if the fractional part is >= 0.5.
    Clamps the result to the valid label range [1, 3].

    With N_RUNS > 1, the mean of repeated calls produces a float whose
    fractional part reflects genuine model disagreement. A value like 1.6
    means the model leaned toward 'medium' more often than not, so we
    promote it rather than rounding to 2 by conventional .round() which
    can mask boundary cases.

    Examples:
        1.4  → 1  (easy,   frac < 0.5, no bump)
        1.5  → 2  (medium, frac >= 0.5, bumped)
        2.49 → 2  (medium, no bump)
        2.5  → 3  (hard,   bumped)
        2.95 → 3  (hard,   bumped, clamped at 3)
    """
    base   = int(score)        # floor (not round)
    frac   = score - base
    result = base + 1 if frac >= 0.5 else base
    return max(1, min(3, result))

# ---------- LOAD CHECKPOINT ----------
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "r") as f:
        saved_scores = json.load(f)
    log.info(f"Loaded {len(saved_scores)} cached scores from checkpoint.")
else:
    saved_scores = {}

scores_raw = []  # float means across N_RUNS, one per row

# ---------- LLM SCORING LOOP ----------
FALLBACK_SCORE = 2.0  # neutral fallback; logged explicitly when used

for idx, row in tqdm(df.iterrows(), total=len(df)):
    key = str(idx)

    if key in saved_scores:
        scores_raw.append(saved_scores[key])
        continue

    run_scores = []
    for run in range(N_RUNS):
        s = llm_score(row["text"], row["code"], idx=f"{idx}-run{run + 1}")
        if s is not None:
            run_scores.append(s)
        time.sleep(0.1)

    if run_scores:
        avg = float(np.mean(run_scores))
        if len(run_scores) < N_RUNS:
            log.warning(
                f"[idx={idx}] Only {len(run_scores)}/{N_RUNS} runs succeeded. "
                f"Mean computed from partial results: {avg:.3f}"
            )
    else:
        log.warning(f"[idx={idx}] All runs failed. Using fallback score {FALLBACK_SCORE}.")
        avg = FALLBACK_SCORE

    scores_raw.append(avg)
    saved_scores[key] = avg

    # Checkpoint every 50 rows
    if idx % 50 == 0:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(saved_scores, f)

# Final checkpoint flush
with open(CHECKPOINT_PATH, "w") as f:
    json.dump(saved_scores, f)
log.info("Checkpoint saved.")

df["llm_score_raw"] = scores_raw  # float mean across runs — used in ensemble blend

# ---------- LOG BUMP RATE (diagnostic) ----------
bumped_count = (df["llm_score_raw"] % 1 >= 0.5).sum()
log.info(
    f"Fractional bump will apply to {bumped_count}/{len(df)} problems "
    f"({bumped_count / len(df) * 100:.1f}%). "
    + ("Consider increasing N_RUNS if this is > 40% with N_RUNS=1." if N_RUNS == 1 else "")
)

# ---------- ENSEMBLE BLEND ----------
# Blend the raw float LLM mean with normalized static features,
# then apply the fractional bump ONCE at the final label step.
# Do NOT apply the bump earlier — it would discard the fractional info
# needed for meaningful blending with static metrics.
df["final_score"] = (
    W_LLM        * df["llm_score_raw"]   +
    W_COMPLEXITY * df["complexity_norm"] +
    W_BRANCHES   * df["branches_norm"]
)

log.info(
    f"Final score stats — mean={df['final_score'].mean():.3f}, "
    f"std={df['final_score'].std():.3f}, "
    f"min={df['final_score'].min():.3f}, "
    f"max={df['final_score'].max():.3f}"
)

# ---------- FINAL DIFFICULTY LABEL ----------
difficulty_map = {1: "easy", 2: "medium", 3: "hard"}

df["difficulty"] = (
    df["final_score"]
    .apply(apply_fractional_bump)
    .map(difficulty_map)
    .fillna("medium")
)

# ---------- AUDIT COLUMN (remove before production if desired) ----------
df["score_breakdown"] = df.apply(
    lambda r: (
        f"llm_raw={r.llm_score_raw:.2f} "
        f"cpx_norm={r.complexity_norm:.2f} "
        f"br_norm={r.branches_norm:.2f} "
        f"→ final={r.final_score:.2f} "
        f"→ {r.difficulty}"
    ),
    axis=1
)

# ---------- COST SUMMARY ----------
total_cost = (
    (total_input_tokens  / 1_000_000 * COST_PER_M_INPUT) +
    (total_output_tokens / 1_000_000 * COST_PER_M_OUTPUT)
)

log.info("=" * 60)
log.info("COST SUMMARY")
log.info(f"  Total API calls:     {total_calls}")
log.info(f"  Failed calls:        {failed_calls}")
log.info(f"  Total input tokens:  {total_input_tokens:,}")
log.info(f"  Total output tokens: {total_output_tokens:,}")
log.info(f"  Input cost:          ${total_input_tokens  / 1_000_000 * COST_PER_M_INPUT:.4f}")
log.info(f"  Output cost:         ${total_output_tokens / 1_000_000 * COST_PER_M_OUTPUT:.4f}")
log.info(f"  TOTAL COST:          ${total_cost:.4f}")
log.info("=" * 60)

# ---------- DISTRIBUTIONS ----------
llm_bumped_dist = (
    df["llm_score_raw"]
    .apply(apply_fractional_bump)
    .value_counts()
    .sort_index()
)
log.info(f"LLM-only (post-bump) distribution:\n{llm_bumped_dist.to_string()}")

final_dist = df["final_score"].apply(apply_fractional_bump).value_counts().sort_index()
log.info(f"Ensemble final score distribution:\n{final_dist.to_string()}")

diff_dist = df["difficulty"].value_counts()
log.info(f"Difficulty label distribution:\n{diff_dist.to_string()}")

# ---------- SAVE ----------
df.to_json(OUTPUT_PATH, orient="records", lines=True)
log.info(f"Saved {len(df)} records to {OUTPUT_PATH}")

summary_cols = [
    "task_id", "llm_score_raw", "complexity", "complexity_norm",
    "branches", "branches_norm", "final_score", "difficulty", "score_breakdown"
]
summary_cols = [c for c in summary_cols if c in df.columns]
log.info(f"Sample output:\n{df[summary_cols].head(10).to_string()}")
log.info(f"Log written to {LOG_PATH}")
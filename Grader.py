"""
grader.py — Two complementary evaluation modes for the Microfinance environment.

1. Programmatic grader   : fast, deterministic, used in Round 1 automated eval
2. LLM grader            : richer qualitative scoring, used in Round 2 / finals

Both return a GradeResult with a 0–1 normalised score and a breakdown dict.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional

# ── Result container ──────────────────────────────────────────────────────

@dataclass
class GradeResult:
    """Unified result from either grader."""
    score: float                             # 0.0 – 1.0 normalised
    passed: bool                             # score >= threshold (0.6)
    breakdown: dict = field(default_factory=dict)
    llm_feedback: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# 1. PROGRAMMATIC GRADER
# ═════════════════════════════════════════════════════════════════════════════

def programmatic_grade(episode_log: dict) -> GradeResult:
    """
    Evaluate one episode from its log dict.

    episode_log keys expected
    ─────────────────────────
    decision          : "APPROVE" | "REJECT" | "TIMEOUT"
    ground_truth      : "APPROVE" | "REJECT"
    default_prob      : float  (0–1, hidden from agent during episode)
    steps_taken       : int
    docs_requested    : list[str]
    cumulative_penalty: float
    terminal_reward   : float
    has_conflicting_signal : bool
    is_borderline          : bool
    """
    decision     = episode_log.get("decision", "TIMEOUT")
    ground_truth = episode_log.get("ground_truth", "REJECT")
    default_prob = episode_log.get("default_prob", 0.5)
    steps        = episode_log.get("steps_taken", 7)
    penalty      = episode_log.get("cumulative_penalty", 0.0)
    terminal_rew = episode_log.get("terminal_reward", -1.5)
    borderline   = episode_log.get("is_borderline", False)

    breakdown = {}

    # ── Correctness (50 % of score) ───────────────────────────────────────
    correct = decision == ground_truth
    breakdown["correct_decision"] = correct

    if correct:
        correctness_score = 1.0
    elif decision == "TIMEOUT":
        correctness_score = 0.0
    else:
        # Wrong but decisive — partial credit based on how ambiguous the case was
        ambiguity = 1.0 - abs(default_prob - 0.5) * 2   # 0 = clear-cut, 1 = 50/50
        correctness_score = ambiguity * 0.3              # max 0.3 for wrong on ambiguous
    breakdown["correctness_score"] = round(correctness_score, 3)

    # ── Efficiency (30 % of score) ────────────────────────────────────────
    # Ideal: decide in ≤3 steps on easy cases, ≤5 on borderline
    ideal_steps = 5 if borderline else 3
    efficiency  = max(0.0, 1.0 - max(0, steps - ideal_steps) / 4)
    breakdown["efficiency_score"]  = round(efficiency, 3)
    breakdown["steps_taken"]       = steps

    # ── Information strategy (20 % of score) ─────────────────────────────
    # Did the agent request relevant docs? Did it avoid redundant requests?
    docs = episode_log.get("docs_requested", [])
    unique_docs   = len(set(docs))
    redundant_req = len(docs) - unique_docs

    # Requesting income proof AND credit history on borderline cases is good
    if borderline:
        info_score = 1.0 if unique_docs >= 2 else 0.5
    else:
        # On easy cases, requesting too many docs is inefficient
        info_score = max(0.0, 1.0 - redundant_req * 0.4 - max(0, unique_docs - 2) * 0.2)

    breakdown["info_strategy_score"] = round(info_score, 3)
    breakdown["docs_requested"]      = docs

    # ── Composite ─────────────────────────────────────────────────────────
    composite = (
        correctness_score * 0.50
        + efficiency       * 0.30
        + info_score       * 0.20
    )
    breakdown["composite_score"]  = round(composite, 3)
    breakdown["terminal_reward"]  = terminal_rew

    return GradeResult(
        score    = round(composite, 4),
        passed   = composite >= 0.60,
        breakdown= breakdown,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. LLM GRADER  (returns the prompt; caller passes it to any LLM API)
# ═════════════════════════════════════════════════════════════════════════════

LLM_GRADER_SYSTEM = """
You are an expert evaluator for a microfinance credit-decision RL environment.
Your job is to assess the QUALITY OF REASONING and DECISION STRATEGY of an AI
agent that processed a loan application.

You will receive:
  - The sequence of actions taken by the agent (with rationales if provided)
  - The applicant profile that was gradually revealed
  - The ground-truth outcome (whether the applicant was a good borrower)

Score the agent on THREE dimensions, each 0–10:

1. REASONING QUALITY
   - Did the agent request documents in a logical order?
   - Did it correctly identify conflicting signals (e.g., high income but many dependents)?
   - Did it handle uncertainty appropriately?

2. DECISION ACCURACY
   - Was the final APPROVE/REJECT consistent with the evidence collected?
   - On borderline cases, did it gather sufficient information before deciding?
   - Did it avoid both false positives (approving defaulters) and false negatives
     (rejecting creditworthy borrowers)?

3. FINANCIAL INCLUSION AWARENESS
   - Did the agent show awareness that over-rejection harms vulnerable borrowers?
   - For rural applicants or seasonal workers, did it avoid purely income-based rejection?
   - Did it consider the full context (repayment streak, community tier, stability)?

Respond ONLY with valid JSON in this exact format:
{
  "reasoning_quality": <int 0-10>,
  "decision_accuracy": <int 0-10>,
  "financial_inclusion": <int 0-10>,
  "weighted_score": <float 0-1>,
  "summary": "<2-3 sentence qualitative assessment>",
  "key_strength": "<one sentence>",
  "key_weakness": "<one sentence>"
}
""".strip()


def build_llm_grader_prompt(episode_log: dict) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) ready to pass to any LLM API.

    episode_log should include the 'action_trace' list, e.g.:
    [
      {"step": 1, "action": "REQUEST_INCOME_PROOF",  "rationale": "..."},
      {"step": 2, "action": "REQUEST_CREDIT_HISTORY","rationale": "..."},
      {"step": 3, "action": "APPROVE",               "rationale": "..."},
    ]
    plus all keys from programmatic_grade()'s episode_log.
    """
    action_trace  = episode_log.get("action_trace", [])
    profile_view  = episode_log.get("revealed_profile", {})
    decision      = episode_log.get("decision", "TIMEOUT")
    ground_truth  = episode_log.get("ground_truth", "UNKNOWN")
    default_prob  = episode_log.get("default_prob", 0.5)
    borderline    = episode_log.get("is_borderline", False)
    conflicting   = episode_log.get("has_conflicting_signal", False)

    user_prompt = f"""
## Applicant Profile (as revealed to agent)

```json
{json.dumps(profile_view, indent=2)}
```

## Agent Action Trace

```json
{json.dumps(action_trace, indent=2)}
```

## Outcome

- Agent decision   : **{decision}**
- Ground truth     : **{ground_truth}**
- Default prob     : {default_prob:.2f}
- Case type        : {"BORDERLINE" if borderline else "CONFLICTING SIGNALS" if conflicting else "STANDARD"}

Evaluate the agent's performance using the three dimensions described in the
system prompt and return valid JSON only.
""".strip()

    return LLM_GRADER_SYSTEM, user_prompt


# ═════════════════════════════════════════════════════════════════════════════
# 3. BATCH EVALUATION HARNESS
# ═════════════════════════════════════════════════════════════════════════════

def batch_evaluate(
    episode_logs: list[dict],
    use_llm: bool = False,
    llm_fn=None,          # callable(system, user) -> str  (JSON response)
) -> dict:
    """
    Run programmatic (and optionally LLM) grading over multiple episodes.
    Returns aggregate statistics.
    """
    results         = [programmatic_grade(log) for log in episode_logs]
    scores          = [r.score for r in results]
    passed          = sum(1 for r in results if r.passed)

    # Breakdown by case type
    borderline_logs  = [l for l in episode_logs if l.get("is_borderline")]
    conflict_logs    = [l for l in episode_logs if l.get("has_conflicting_signal")]

    def _avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = {
        "n_episodes"         : len(episode_logs),
        "pass_rate"          : round(passed / max(len(results), 1), 4),
        "mean_score"         : _avg(scores),
        "median_score"       : _avg(sorted(scores)[len(scores) // 2 : len(scores) // 2 + 1]),
        "borderline_accuracy": _avg([
            1.0 if episode_logs[i].get("decision") == episode_logs[i].get("ground_truth") else 0.0
            for i in range(len(episode_logs)) if episode_logs[i].get("is_borderline")
        ]),
        "conflict_accuracy"  : _avg([
            1.0 if episode_logs[i].get("decision") == episode_logs[i].get("ground_truth") else 0.0
            for i in range(len(episode_logs)) if episode_logs[i].get("has_conflicting_signal")
        ]),
        "mean_steps"         : _avg([l.get("steps_taken", 7) for l in episode_logs]),
    }

    if use_llm and llm_fn:
        llm_scores = []
        for log in episode_logs[:20]:   # cap LLM calls at 20 to save tokens
            sys_p, usr_p = build_llm_grader_prompt(log)
            try:
                raw  = llm_fn(sys_p, usr_p)
                data = json.loads(raw)
                llm_scores.append(data.get("weighted_score", 0.0))
            except Exception:
                pass
        summary["llm_mean_score"] = _avg(llm_scores)

    return summary
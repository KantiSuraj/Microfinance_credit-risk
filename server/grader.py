"""
grader.py — Trajectory-aware evaluation for the two-phase environment (v2).

Unlike v1, this grader evaluates the FULL TRAJECTORY across both phases
with anti-reward-hacking hardening:

1. Phase 1 quality      (30%): Was approval/rejection correct? Efficient?
2. Phase 2 quality      (35%): Were interventions well-timed? Adapted to noise?
3. Timing score         (10%): Did early vs late intervention matter?
4. Information flow     (10%): Did Phase 1 doc collection improve Phase 2 outcomes?
5. Information sufficiency (15%) [NEW]: Did agent gather enough evidence before deciding?

v2 anti-hacking changes:
  ✓ Info sufficiency dimension — blind decisions heavily penalized
  ✓ Efficiency U-curve — too FEW steps is also bad (not just too many)
  ✓ Phase 2 score for rejects capped — can't skip Phase 2 and get full credit
  ✓ Counterfactual integration — soft penalty for high-confidence wrong decisions
  ✓ Over-intervention detection — diminishing returns on restructures
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional, List

import sys, os
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.counterfactual import counterfactual_grade_modifier


@dataclass
class TrajectoryGrade:
    score             : float
    passed            : bool
    phase1_score      : float
    phase2_score      : float
    timing_score      : float
    info_flow_score   : float
    info_sufficiency  : float = 0.5
    breakdown         : dict = field(default_factory=dict)
    llm_feedback      : Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════
# PROGRAMMATIC TRAJECTORY GRADER (v2)
# ══════════════════════════════════════════════════════════════════════════

def grade_trajectory(episode_log: dict) -> TrajectoryGrade:
    """
    Full trajectory evaluation with anti-reward-hacking hardening.

    episode_log expected keys
    ─────────────────────────
    phase1_decision        : "APPROVE" | "REJECT"
    ground_truth           : "APPROVE" | "REJECT"
    default_prob           : float
    phase1_steps           : int
    docs_collected         : list[str]
    phase1_reward          : float
    reached_phase2         : bool
    terminal_outcome       : "REPAID" | "DEFAULT" | "ESCALATED" | "TIMEOUT"
    terminal_reward        : float
    payment_history        : list["ON_TIME" | "MISSED"]  (true values)
    intervention_history   : list[str]  e.g. ["M3:REMINDER", "M6:RESTRUCTURE"]
    signal_quality         : float
    is_borderline          : bool
    has_conflicting_signal : bool
    default_prob_at_month3 : float  (actual hidden prob at month 3)
    default_prob_at_month6 : float  (actual hidden prob at month 6)
    """
    bd = {}   # breakdown accumulator

    p1_decision  = episode_log.get("phase1_decision", "REJECT")
    ground_truth = episode_log.get("ground_truth", "REJECT")
    default_prob = episode_log.get("default_prob", 0.5)
    p1_steps     = episode_log.get("phase1_steps", 7)
    docs         = episode_log.get("docs_collected", [])
    phase2       = episode_log.get("reached_phase2", False)
    outcome      = episode_log.get("terminal_outcome", "TIMEOUT")
    p1_reward    = episode_log.get("phase1_reward", -1.5)
    t_reward     = episode_log.get("terminal_reward", -1.5)
    payments     = episode_log.get("payment_history", [])
    interventions= episode_log.get("intervention_history", [])
    sq           = episode_log.get("signal_quality", 0.60)
    borderline   = episode_log.get("is_borderline", False)
    conflicting  = episode_log.get("has_conflicting_signal", False)

    n_docs = len(docs)

    # ── Phase 1 score (30 %) ──────────────────────────────────────────────
    correct = (p1_decision == ground_truth)
    bd["phase1_correct"] = correct

    if correct:
        p1_correctness = 1.0
    elif outcome in ("REPAID",):
        # Wrong approval but loan repaid — outcome redeems the mistake partially
        p1_correctness = 0.4
    elif p1_decision == "APPROVE" and outcome == "DEFAULT":
        p1_correctness = 0.0
    else:
        ambiguity = 1.0 - abs(default_prob - 0.5) * 2
        p1_correctness = ambiguity * 0.25

    # ── Efficiency U-curve (v2) ───────────────────────────────────────────
    # Too FEW steps (deciding with no investigation) is also bad.
    # Ideal: 2-3 steps for clear cases, 4-5 for borderline.
    # 1 step with 0 docs → low efficiency (rushing)
    # exact ideal steps → peak efficiency
    # too many steps → also low (over-investigating)
    ideal_p1_steps = 5 if borderline else 3
    if p1_steps <= 1 and n_docs == 0:
        # Rushed decision with zero investigation — very low efficiency
        p1_efficiency = 0.20
    elif p1_steps <= 1 and n_docs > 0:
        # Quick but gathered at least some info — moderate
        p1_efficiency = 0.55
    else:
        # Standard efficiency: penalize for going over ideal
        over = max(0, p1_steps - ideal_p1_steps)
        under = max(0, ideal_p1_steps - p1_steps - 1)  # -1 so being 1 under is fine
        p1_efficiency = max(0.0, 1.0 - over / 4 - under / 6)

    phase1_score = p1_correctness * 0.7 + p1_efficiency * 0.3
    bd["phase1_correctness"] = round(p1_correctness, 3)
    bd["phase1_efficiency"]  = round(p1_efficiency, 3)
    bd["phase1_score"]       = round(phase1_score, 3)

    # ── Phase 2 score (35 %) ──────────────────────────────────────────────
    if not phase2:
        # Episode ended in Phase 1 (rejected or timeout)
        # v2: DON'T give full credit for skipping Phase 2
        # A reject means the agent avoided the harder challenge entirely.
        # Cap at 0.5 × p1_correctness — agent didn't prove Phase 2 skill.
        phase2_score = 0.50 * p1_correctness
        bd["phase2_note"] = "No Phase 2 (rejected/timeout) — capped at 50% of p1 correctness"
    else:
        if outcome == "REPAID":
            outcome_score = 1.0
        elif outcome == "DEFAULT":
            outcome_score = 0.0
        elif outcome == "ESCALATED":
            # Partial credit — agent recognised risk and cut losses
            outcome_score = 0.4
        else:
            outcome_score = 0.1

        # Penalise unnecessary restructures (over-intervention)
        n_restructure = sum(1 for i in interventions if "RESTRUCTURE" in i)
        over_intervene_penalty = max(0.0, (n_restructure - 1) * 0.15)

        phase2_score = max(0.0, outcome_score - over_intervene_penalty)
        bd["phase2_outcome_score"]    = round(outcome_score, 3)
        bd["over_intervene_penalty"]  = round(over_intervene_penalty, 3)
        bd["phase2_score"]            = round(phase2_score, 3)

    # ── Timing score (10 %) ───────────────────────────────────────────────
    # Did the agent intervene BEFORE the borrower deteriorated badly?
    timing_score = 0.5   # neutral default

    if phase2 and interventions:
        # Extract month numbers of first intervention
        first_month = _extract_month(interventions[0])
        n_misses_before_first = sum(
            1 for i, p in enumerate(payments)
            if i < first_month - 1 and p == "MISSED"
        )
        # Ideal: intervene within 2 months of first miss
        if n_misses_before_first == 0:
            timing_score = 1.0   # proactive — intervened before any miss
        elif n_misses_before_first <= 1:
            timing_score = 0.85  # reactive but fast
        elif n_misses_before_first <= 2:
            timing_score = 0.6   # late
        else:
            timing_score = 0.2   # very late (reactive too slow)

        # Bonus: did restructure happen early enough to actually help?
        p_m3 = episode_log.get("default_prob_at_month3", default_prob)
        p_m6 = episode_log.get("default_prob_at_month6", default_prob)
        if p_m3 < p_m6:
            timing_score = max(0.0, timing_score - 0.2)  # interventions too late
    elif phase2 and not interventions:
        # Agent never intervened in Phase 2 — penalise if outcome was default
        timing_score = 0.3 if outcome == "DEFAULT" else 0.7

    bd["timing_score"] = round(timing_score, 3)

    # ── Information flow score (10 %) ─────────────────────────────────────
    # Did collecting more info in Phase 1 correlate with better Phase 2 outcome?
    if sq is None:
        info_flow = 0.7 if docs else 0.5
    elif sq >= 0.90:
        info_flow = 1.0
    elif sq >= 0.75:
        info_flow = 0.7
    else:
        info_flow = 0.4

    # If signal was noisy but agent still got good outcome, partial credit
    if sq is not None and sq < 0.75 and outcome == "REPAID":
        info_flow = min(info_flow + 0.2, 1.0)

    bd["info_flow_score"] = round(info_flow, 3)
    bd["signal_quality"]  = sq

    # ── Information sufficiency score (15 %) [NEW in v2] ──────────────────
    # Penalizes decisions made with inadequate evidence.
    # A well-informed agent should gather at least 1 doc before deciding.
    info_suff = _info_sufficiency_score(n_docs, p1_decision, borderline, conflicting)
    bd["info_sufficiency"] = round(info_suff, 3)

    # ── Counterfactual modifier (soft, baked into composite) ──────────────
    # Only applied when agent had enough info to make an informed choice.
    from reward_engine import info_confidence
    agent_conf = info_confidence(
        "income_proof" in docs or any("income" in d for d in docs),
        "credit_history" in docs or any("credit" in d for d in docs),
    )
    cf_modifier = counterfactual_grade_modifier(
        p1_decision, agent_conf, ground_truth, default_prob, borderline,
    )
    bd["counterfactual_modifier"] = round(cf_modifier, 3)

    # ── Composite (v2 weights) ────────────────────────────────────────────
    # phase1=30%, phase2=35%, timing=10%, info_flow=10%, info_sufficiency=15%
    raw_composite = (
        phase1_score  * 0.30
        + phase2_score  * 0.35
        + timing_score  * 0.10
        + info_flow     * 0.10
        + info_suff     * 0.15
    )

    # Apply counterfactual as a SOFT modifier (10% influence)
    # composite = 90% raw + 10% counterfactual-modified raw
    composite = raw_composite * 0.90 + raw_composite * cf_modifier * 0.10

    composite = max(0.0, min(1.0, composite))  # clamp
    bd["raw_composite"] = round(raw_composite, 4)
    bd["composite"] = round(composite, 4)
    bd["terminal_reward"] = t_reward

    return TrajectoryGrade(
        score            = round(composite, 4),
        passed           = composite >= 0.60,
        phase1_score     = round(phase1_score, 3),
        phase2_score     = round(phase2_score, 3),
        timing_score     = round(timing_score, 3),
        info_flow_score  = round(info_flow, 3),
        info_sufficiency = round(info_suff, 3),
        breakdown        = bd,
    )


def _info_sufficiency_score(
    n_docs: int,
    decision: str,
    borderline: bool,
    conflicting: bool,
) -> float:
    """
    Score for information sufficiency — how well-informed was the decision?

    Returns [0.0, 1.0]:
      0 docs → 0.15 (blind decision — very low)
      1 doc, clear case → 0.75 (targeted — good)
      1 doc, borderline → 0.50 (borderline needs more)
      2 docs → 0.90 (thorough)
      3+ docs → 0.55 (over-requesting — significant diminishing returns)
    """
    if n_docs == 0:
        return 0.15   # blind decision
    elif n_docs == 1:
        if borderline or conflicting:
            return 0.50   # borderline/conflicting needs more investigation
        return 0.75       # clear case, targeted investigation
    elif n_docs == 2:
        return 0.90       # thorough
    else:
        return 0.55       # v2.1: over-requesting penalized harder


def _extract_month(intervention_str: str) -> int:
    """Parse month number from e.g. 'M3:REMINDER' → 3."""
    try:
        return int(intervention_str.split(":")[0][1:])
    except (IndexError, ValueError):
        return 99


# ══════════════════════════════════════════════════════════════════════════
# LLM GRADER PROMPT
# ══════════════════════════════════════════════════════════════════════════

LLM_GRADER_SYSTEM = """
You are an expert evaluator for a two-phase microfinance RL environment.
The agent must: (1) decide whether to approve a loan with partial information,
then (2) monitor the loan monthly and intervene to prevent default.

Score on FOUR dimensions (0–10 each):

1. PHASE 1 REASONING
   Was the approval/rejection decision sound given collected evidence?
   Did the agent request the right documents without over-requesting?

2. PHASE 2 MANAGEMENT
   Were monthly observations interpreted correctly despite noise?
   Was intervention timing appropriate (early vs late)?
   Was the right intervention type chosen (reminder vs restructure)?

3. TRAJECTORY COHERENCE
   Did the agent's Phase 1 information gathering improve Phase 2 decisions?
   Is there evidence the agent traded off Phase 1 cost against Phase 2 signal quality?

4. FINANCIAL INCLUSION AWARENESS
   Did the agent avoid over-rejection of creditworthy borrowers?
   On repaid loans, did the agent avoid over-intervention (unnecessary restructures)?

Respond ONLY with valid JSON:
{
  "phase1_reasoning": <0-10>,
  "phase2_management": <0-10>,
  "trajectory_coherence": <0-10>,
  "financial_inclusion": <0-10>,
  "weighted_score": <0.0-1.0>,
  "summary": "<2-3 sentences>",
  "key_strength": "<one sentence>",
  "key_weakness": "<one sentence>"
}
""".strip()


def build_llm_prompt(episode_log: dict) -> tuple[str, str]:
    user = f"""
## Full Trajectory Log

### Phase 1 — Application
- Decision     : {episode_log.get('phase1_decision')}
- Ground truth : {episode_log.get('ground_truth')}
- Default prob : {episode_log.get('default_prob', 0):.2f}
- Steps taken  : {episode_log.get('phase1_steps')}
- Docs collected: {episode_log.get('docs_collected', [])}
- Signal quality set: {episode_log.get('signal_quality', 'N/A')}

### Phase 2 — Monitoring (if reached)
- Payment history (true): {episode_log.get('payment_history', [])}
- Interventions (timing): {episode_log.get('intervention_history', [])}
- Terminal outcome       : {episode_log.get('terminal_outcome')}
- Terminal reward        : {episode_log.get('terminal_reward')}

### Case metadata
- Borderline case         : {episode_log.get('is_borderline')}
- Has conflicting signals : {episode_log.get('has_conflicting_signal')}

Evaluate this trajectory across all four dimensions and return JSON only.
""".strip()
    return LLM_GRADER_SYSTEM, user


# ══════════════════════════════════════════════════════════════════════════
# ALIASES & BATCH UTILITIES (used by test_env.py)
# ══════════════════════════════════════════════════════════════════════════

# Alias so test_env.py can `from server.grader import programmatic_grade`
programmatic_grade = grade_trajectory


def batch_evaluate(episode_logs: List[dict]) -> dict:
    """
    Evaluate a batch of episode logs and return aggregate statistics.

    Returns a dict with:
      n_episodes, mean_score, median_score, min_score, max_score,
      pass_rate, mean_phase1, mean_phase2, mean_timing, mean_info_flow, mean_info_suff
    """
    grades = [grade_trajectory(log) for log in episode_logs]
    scores = [g.score for g in grades]
    n = len(scores)
    if n == 0:
        return {"n_episodes": 0}

    sorted_scores = sorted(scores)
    median = (
        sorted_scores[n // 2]
        if n % 2 == 1
        else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    )

    return {
        "n_episodes":     n,
        "mean_score":     round(sum(scores) / n, 4),
        "median_score":   round(median, 4),
        "min_score":      round(min(scores), 4),
        "max_score":      round(max(scores), 4),
        "pass_rate":      round(sum(1 for g in grades if g.passed) / n, 3),
        "mean_phase1":    round(sum(g.phase1_score for g in grades) / n, 4),
        "mean_phase2":    round(sum(g.phase2_score for g in grades) / n, 4),
        "mean_timing":    round(sum(g.timing_score for g in grades) / n, 4),
        "mean_info_flow": round(sum(g.info_flow_score for g in grades) / n, 4),
        "mean_info_suff": round(sum(g.info_sufficiency for g in grades) / n, 4),
    }
"""
grader.py — Trajectory-aware evaluation for the two-phase environment.

Unlike v1, this grader evaluates the FULL TRAJECTORY across both phases:

1. Phase 1 quality    : Was approval/rejection correct? Efficient?
2. Phase 2 quality    : Were interventions well-timed? Did the agent
                        adapt to noisy signals appropriately?
3. Information flow   : Did Phase 1 doc collection improve Phase 2 outcomes?
4. Compounding effect : Did early interventions outperform late ones?
                        (measured by comparing default_prob at month 3 vs 6)

The key metric that's impossible for a classifier: TIMING SCORE.
A correct approval followed by a late intervention on a deteriorating
borrower scores lower than the same approval with a timely restructure.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrajectoryGrade:
    score          : float
    passed         : bool
    phase1_score   : float
    phase2_score   : float
    timing_score   : float
    info_flow_score: float
    breakdown      : dict = field(default_factory=dict)
    llm_feedback   : Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════
# PROGRAMMATIC TRAJECTORY GRADER
# ══════════════════════════════════════════════════════════════════════════

def grade_trajectory(episode_log: dict) -> TrajectoryGrade:
    """
    Full trajectory evaluation.

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

    # ── Phase 1 score (35 %) ──────────────────────────────────────────────
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

    ideal_p1_steps = 5 if borderline else 3
    p1_efficiency = max(0.0, 1.0 - max(0, p1_steps - ideal_p1_steps) / 4)
    phase1_score = p1_correctness * 0.7 + p1_efficiency * 0.3
    bd["phase1_correctness"] = round(p1_correctness, 3)
    bd["phase1_efficiency"]  = round(p1_efficiency, 3)
    bd["phase1_score"]       = round(phase1_score, 3)

    # ── Phase 2 score (35 %) ──────────────────────────────────────────────
    if not phase2:
        # Episode ended in Phase 1 — score purely on outcome quality
        phase2_score = p1_correctness   # consistent with Phase 1 judgement
        bd["phase2_note"] = "No Phase 2 (rejected or timeout in Phase 1)"
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

    # ── Timing score (20 %) ───────────────────────────────────────────────
    # The key metric a classifier can never optimise.
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
            timing_score = max(0.0, timing_score - 0.2)  # interventions too late to bend curve
    elif phase2 and not interventions:
        # Agent never intervened in Phase 2 — penalise if outcome was default
        timing_score = 0.3 if outcome == "DEFAULT" else 0.7

    bd["timing_score"] = round(timing_score, 3)

    # ── Information flow score (10 %) ─────────────────────────────────────
    # Did collecting more info in Phase 1 correlate with better Phase 2 outcome?
    # Proxy: signal_quality correlates with ability to catch early warning signs
    if sq is None:
        # Rejected in Phase 1 — no Phase 2 signal to evaluate
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

    # ── Composite ─────────────────────────────────────────────────────────
    composite = (
        phase1_score  * 0.35
        + phase2_score  * 0.35
        + timing_score  * 0.20
        + info_flow     * 0.10
    )
    bd["composite"] = round(composite, 4)
    bd["terminal_reward"] = t_reward

    return TrajectoryGrade(
        score           = round(composite, 4),
        passed          = composite >= 0.60,
        phase1_score    = round(phase1_score, 3),
        phase2_score    = round(phase2_score, 3),
        timing_score    = round(timing_score, 3),
        info_flow_score = round(info_flow, 3),
        breakdown       = bd,
    )


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
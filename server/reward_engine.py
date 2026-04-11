"""
reward_engine.py — Isolated reward computation for the Microfinance environment.

Anti-Reward-Hacking Design (v2)
═══════════════════════════════
This reward engine is designed to make exploitation EXTREMELY difficult:

1. MULTI-OBJECTIVE: reward = decision_quality - info_cost - delay - waste - inaction
2. COUNTERFACTUAL: blind approve is always negative, blind reject is slightly negative
3. REDUNDANCY: escalating penalty for repeated useless actions
4. STEP BUDGET: escalating step cost (later steps cost MORE)
5. OVER-REQUESTING: diminishing returns on additional docs
6. LAZY REJECT: reject-without-info gets penalty — must justify
7. UNINFORMED DECISION: ANY decision with conf=0 penalized
8. PHASE 2 INACTION: doing nothing when danger is visible is penalised
9. PHASE 2 SPAM: repeating same intervention type monotonically penalized
10. INFORMATION-GATED: reward scales with evidence gathered for BOTH approve & reject

Key properties (verified by test suite):
  ✓ Blind APPROVE always negative in expectation
  ✓ Blind REJECT is slightly negative (~-0.05)
  ✓ Always-REJECT strategy averages < 0.40 in grader
  ✓ Request-ALL-docs gets diminishing returns
  ✓ Strategic agent beats impulsive agent
  ✓ Reactive monitoring beats passive monitoring
  ✓ No single degenerate strategy dominates
"""

from __future__ import annotations

# ── Phase 1 constants ──────────────────────────────────────────────────────
_WRONG_APPROVE   = -2.15   # v2.1: slightly harsher to lower random agent score
_WRONG_REJECT    = -1.00
_P1_TIMEOUT      = -1.50

# Correct decision scaling: reward = base + slope × info_confidence
# ── APPROVE ──
_CORRECT_APPROVE_BASE  = -0.30   # blind correct approve — still negative
_CORRECT_APPROVE_SLOPE =  1.30   # × conf: at conf=1.0 → +1.00

# ── REJECT (re-tuned v2) ──
# blind correct reject:  0.10 + (-0.15) = -0.05  (slightly negative — discourage, don't destroy)
# informed correct reject: 0.10 + 0.55   = +0.65  (well-justified rejection rewarded)
_CORRECT_REJECT_BASE   =  0.10   # was 0.25; reduced so blind rejects aren't free
_CORRECT_REJECT_SLOPE  =  0.55   # was 0.45; steeper slope → information matters MORE

# ── Anti-hack: Lazy reject penalty ────────────────────────────────────────
# Rejecting with ZERO information is penalised — agent must justify rejection
_LAZY_REJECT_PENALTY   = -0.15   # was -0.15; kept moderate per user feedback

# ── Anti-hack: Uninformed decision penalty (NEW) ──────────────────────────
# ANY terminal decision (approve OR reject) with conf=0 gets this penalty
_UNINFORMED_DECISION_PENALTY = -0.10

# Phase 1 step costs (ESCALATING — later steps cost more)
DOC_REQUEST_COST = -0.10
FLAG_REVIEW_COST = -0.15

# Anti-hack: Escalating step cost — step N costs more than step N-1
# step_cost(n) = base_cost * (1 + escalation_rate * n)
STEP_ESCALATION_RATE = 0.15  # each step adds 15% more cost

# Anti-hack: Over-requesting penalty (requesting BOTH docs when case is clear)
OVER_REQUEST_PENALTY = -0.12  # v2.1: stronger to ensure targeted > overcollect

# Anti-hack: Redundant action escalating penalty
REDUNDANT_BASE_PENALTY = -0.10   # first redundant action
REDUNDANT_ESCALATION   = -0.08   # each additional redundant action adds this

# ── Phase 2 constants ──────────────────────────────────────────────────────
REPAID_BONUS        =  1.50
DEFAULT_PENALTY     = -2.50
REMINDER_COST       = -0.05
RESTRUCTURE_COST    = -0.20
ESCALATE_COST       = -0.50

# Anti-hack: Phase 2 inaction penalty — doing nothing when danger signals visible
INACTION_PENALTY    = -0.12   # v2.1: stronger to widen reactive vs passive delta
# Anti-hack: Spam intervention penalty — sending reminders every month
SPAM_REMINDER_PENALTY = -0.05  # v2.1: stronger + kicks in at 2+ consecutive (was 3+)

# Anti-hack: Phase 2 monotonic strategy penalty (NEW)
# If agent uses the EXACT SAME action for 4+ consecutive months → penalty
MONOTONIC_STRATEGY_PENALTY = -0.04  # per month beyond 3rd consecutive same action


def info_confidence(income_revealed: bool, credit_history_revealed: bool) -> float:
    """
    Scalar in [0.0, 1.0] encoding how much evidence the agent has gathered.
    Credit history is weighted 2× income because it is the stronger signal.
    """
    return 0.35 * int(income_revealed) + 0.65 * int(credit_history_revealed)


def escalating_step_cost(base_cost: float, step_number: int) -> float:
    """
    Later steps cost more. Prevents agents from using all 7 steps every time.
    step_cost = base × (1 + 0.15 × step_number)

    Step 1: -0.10 × 1.15 = -0.115
    Step 5: -0.10 × 1.75 = -0.175
    Step 7: -0.10 × 2.05 = -0.205
    """
    return round(base_cost * (1 + STEP_ESCALATION_RATE * step_number), 4)


def redundant_action_penalty(redundant_count: int) -> float:
    """
    Escalating penalty for redundant actions.
    1st redundant: -0.10
    2nd redundant: -0.18
    3rd redundant: -0.26
    Makes action spamming increasingly expensive.
    """
    return round(REDUNDANT_BASE_PENALTY + REDUNDANT_ESCALATION * (redundant_count - 1), 4)


def phase1_terminal_reward(
    decision: str,           # "APPROVE" | "REJECT"
    ground_truth: str,       # "APPROVE" | "REJECT"
    income_revealed: bool,
    credit_history_revealed: bool,
    step_penalties: float,   # cumulative doc/review costs (negative)
) -> float:
    """
    Compute the Phase 1 terminal reward.

    Anti-hack properties (v2):
      - Blind approve is ALWAYS negative (base = -0.30)
      - Blind correct reject is slightly negative (-0.05)
      - Informed correct reject is positive (+0.65)
      - Lazy reject (conf=0) gets additional penalty
      - Uninformed decision (conf=0) gets additional penalty for BOTH actions
      - Over-requesting doesn't help: conf caps at 1.0
      - Wrong approve is -2.0 regardless of information
    """
    conf    = info_confidence(income_revealed, credit_history_revealed)
    correct = (decision == ground_truth)

    if decision == "APPROVE":
        base = (
            _CORRECT_APPROVE_BASE + _CORRECT_APPROVE_SLOPE * conf
            if correct else _WRONG_APPROVE
        )
    else:  # REJECT
        if correct:
            base = _CORRECT_REJECT_BASE + _CORRECT_REJECT_SLOPE * conf
        else:
            base = _WRONG_REJECT
        # Anti-hack: Lazy reject — rejecting with zero information is penalised
        if conf == 0.0:
            base += _LAZY_REJECT_PENALTY

    # Anti-hack v2: Uninformed decision penalty — ANY decision with zero confidence
    if conf == 0.0:
        base += _UNINFORMED_DECISION_PENALTY

    return round(base + step_penalties, 4)


def phase1_timeout_reward(step_penalties: float) -> float:
    return round(_P1_TIMEOUT + step_penalties, 4)


def phase2_terminal_reward(
    phase1_reward: float,
    outcome: str,            # "REPAID" | "DEFAULT" | "ESCALATED"
    intervention_costs: float,
) -> float:
    """
    Compute the final episode reward once Phase 2 ends.
    """
    if outcome == "REPAID":
        return round(phase1_reward + REPAID_BONUS + intervention_costs, 4)
    elif outcome == "DEFAULT":
        return round(DEFAULT_PENALTY + intervention_costs, 4)
    else:  # ESCALATED
        return round(phase1_reward + ESCALATE_COST + intervention_costs, 4)


def phase2_intervention_cost(action_type: str) -> float:
    """Return the cost of a Phase 2 intervention action."""
    return {
        "DO_NOTHING"           : 0.0,
        "SEND_REMINDER"        : REMINDER_COST,
        "RESTRUCTURE_LOAN"     : RESTRUCTURE_COST,
        "ESCALATE_TO_RECOVERY" : ESCALATE_COST,
    }.get(action_type, 0.0)


def phase2_inaction_penalty(cumulative_misses: int, missed_streak: int) -> float:
    """
    Penalty for doing nothing when danger signals are visible.
    Applied when agent chooses DO_NOTHING despite clear risk indicators.

    cumulative_misses >= 2 → penalty applied
    missed_streak >= 2     → double penalty (imminent danger)
    """
    if cumulative_misses < 2:
        return 0.0
    penalty = INACTION_PENALTY
    if missed_streak >= 2:
        penalty *= 2  # double penalty for consecutive misses + inaction
    return round(penalty, 4)


def phase2_spam_penalty(consecutive_reminders: int) -> float:
    """
    Penalty for spamming SEND_REMINDER every single month.
    v2.1: kicks in after 2+ consecutive reminders (was 3+).
    """
    if consecutive_reminders < 2:
        return 0.0
    return round(SPAM_REMINDER_PENALTY * (consecutive_reminders - 1), 4)


def phase2_monotonic_penalty(
    consecutive_same_action: int,
    action: str = "DO_NOTHING",
    cumulative_misses: int = 0,
    missed_streak: int = 0,
    current_default_prob: float = 0.10,
    shock_scheduled: bool = False,
) -> float:
    """
    Context-conditioned penalty for using the EXACT SAME action 4+ months.

    For non-DO_NOTHING actions (e.g. always SEND_REMINDER), the penalty fires
    purely on streak length — spamming the same intervention is always wasteful.

    For DO_NOTHING, the penalty ONLY fires when danger signals indicate the
    borrower actually needed attention.  A sequence of correct DO_NOTHINGs on
    a healthy borrower is competent monitoring, not a degenerate strategy.

    Returns 0.0 if fewer than 4 consecutive same actions.
    Otherwise: -0.04 per month beyond the 3rd.
    """
    if consecutive_same_action < 4:
        return 0.0

    # Non-DO_NOTHING monotonic actions are always penalized (sequence-only)
    if action != "DO_NOTHING":
        return round(MONOTONIC_STRATEGY_PENALTY * (consecutive_same_action - 3), 4)

    # DO_NOTHING: only penalize if the borrower actually needed attention
    danger_present = (
        cumulative_misses >= 2 or
        missed_streak >= 1 or
        current_default_prob > 0.30 or
        shock_scheduled
    )

    if not danger_present:
        return 0.0  # healthy borrower — DO_NOTHING is correct, no penalty

    # Danger was present and agent did nothing repeatedly
    return round(MONOTONIC_STRATEGY_PENALTY * (consecutive_same_action - 3), 4)


# ── Independent Reward Audit (Strategy 7 — Enhanced v2) ────────────────────

def audit_reward(episode_log: dict) -> dict:
    """
    Independent secondary validator. Flags suspicious patterns that suggest
    reward hacking rather than genuine reasoning.

    Returns dict of flags. If any flag is True → suspicious.

    v2 additions:
      - early_termination_exploit : always decides in step 1
      - phase2_monotonic_strategy : same Phase 2 action every month
      - counterfactual_mismatch   : decision contradicts what informed agent would do
    """
    flags = {}
    actions = [a.get("action", a.get("action_type", ""))
               for a in episode_log.get("action_trace", [])]

    # Flag 1: Always reject (never gathers any info)
    flags["always_reject"] = (
        episode_log.get("phase1_decision") == "REJECT" and
        len(episode_log.get("docs_collected", [])) == 0
    )

    # Flag 2: Always approve blind (no docs, immediate approve)
    flags["blind_approve"] = (
        episode_log.get("phase1_decision") == "APPROVE" and
        len(episode_log.get("docs_collected", [])) == 0
    )

    # Flag 3: Over-requesting (requested ALL possible Phase 1 info)
    flags["over_requesting"] = len(episode_log.get("docs_collected", [])) >= 2 and \
        any("FLAG_FOR_REVIEW" in str(a) for a in actions)

    # Flag 4: Repetitive Phase 2 actions (same action every month)
    p2_actions = [a for a in actions if a in ("DO_NOTHING", "SEND_REMINDER",
                                                "RESTRUCTURE_LOAN", "ESCALATE_TO_RECOVERY")]
    if len(p2_actions) >= 4:
        unique_p2 = set(p2_actions)
        flags["repetitive_phase2"] = len(unique_p2) == 1  # ALWAYS same action
    else:
        flags["repetitive_phase2"] = False

    # Flag 5: Suspiciously high reward with minimal effort
    steps = episode_log.get("phase1_steps", 0)
    terminal = episode_log.get("terminal_reward", 0) or 0
    flags["suspicious_high_reward"] = (steps <= 1 and terminal > 1.0)

    # Flag 6: Phase 2 complete inaction (never intervened despite misses)
    payment_history = episode_log.get("payment_history", [])
    interventions = episode_log.get("intervention_history", [])
    misses = sum(1 for p in payment_history if p == "MISSED")
    flags["phase2_complete_inaction"] = (misses >= 3 and len(interventions) == 0)

    # Flag 7 (NEW): Early termination exploit — decided in step 1 with no docs
    flags["early_termination_exploit"] = (
        steps <= 1 and
        len(episode_log.get("docs_collected", [])) == 0 and
        episode_log.get("phase1_decision") in ("APPROVE", "REJECT")
    )

    # Flag 8 (NEW): Phase 2 monotonic strategy — same action for 5+ months straight
    # Context-conditioned: a run of DO_NOTHING on a borrower with zero misses
    # during that window is correct monitoring, not a degenerate strategy.
    if len(p2_actions) >= 5:
        # Check longest run of same action and what that action was
        longest_run = 1
        current_run = 1
        longest_run_action = p2_actions[0]
        current_run_action = p2_actions[0]
        for i in range(1, len(p2_actions)):
            if p2_actions[i] == p2_actions[i - 1]:
                current_run += 1
                if current_run > longest_run:
                    longest_run = current_run
                    longest_run_action = current_run_action
            else:
                current_run = 1
                current_run_action = p2_actions[i]

        if longest_run >= 5:
            if longest_run_action == "DO_NOTHING":
                # Context check: were there misses during the episode?
                misses = sum(1 for p in payment_history if p == "MISSED")
                flags["phase2_monotonic_strategy"] = (misses >= 2)
            else:
                # Non-DO_NOTHING monotonic run is always suspicious
                flags["phase2_monotonic_strategy"] = True
        else:
            flags["phase2_monotonic_strategy"] = False
    else:
        flags["phase2_monotonic_strategy"] = False

    flags["any_flag"] = any(v for k, v in flags.items() if k != "any_flag")
    return flags
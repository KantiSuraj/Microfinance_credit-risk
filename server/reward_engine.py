"""
reward_engine.py — Isolated reward computation for the Microfinance environment.

Design principle (from checklist):
  "Don't mix logic inside step(). Reward isolation = separate function/module."

The central insight that makes this environment non-trivial:
  Correct decisions made WITHOUT information are penalised.
  The agent cannot get full reward by guessing right — it must EARN
  the decision by gathering evidence first.

Information confidence
──────────────────────
  conf = 0.35 × income_revealed + 0.65 × credit_history_revealed

  Credit history is weighted higher because it directly encodes past
  repayment behaviour, which is the strongest predictor of future default.
  Income alone can be misleading (conflicting signals).

Phase 1 reward landscape
────────────────────────
  CORRECT APPROVE
    conf=0.00 (blind)          → -0.30   ← always negative, agent can't win blind
    conf=0.35 (income only)    → +0.155  ← barely profitable
    conf=0.65 (credit only)    → +0.545
    conf=1.00 (both docs)      → +1.000  ← maximum

  WRONG APPROVE (false positive)
    always -2.0                 ← harsh, no information mercy

  CORRECT REJECT
    conf=0.00 (blind)          → +0.25   ← conservative reject is defensible
    conf=1.00 (both docs)      → +0.70   ← agent confirmed the reject properly

  WRONG REJECT (false negative, financial exclusion harm)
    always -1.0                 ← meaningful penalty

  This asymmetry means:
  - Blind APPROVE is always a losing strategy (expected reward ≈ -1.1)
  - Blind REJECT is low-reward but not catastrophic
  - Gathering information is necessary for high-reward APPROVE decisions
  - The agent must decide when information cost outweighs information gain

Phase 2 reward landscape
────────────────────────
  REPAID (after full 12 months)   : +1.5 bonus added to Phase 1 reward
  DEFAULT (cumulative misses ≥ 4) : -2.5, replaces approval reward entirely
  SEND_REMINDER                   : -0.05 per use
  RESTRUCTURE_LOAN                : -0.20 per use
  ESCALATE_TO_RECOVERY            : -0.50, episode ends
  Max steps timeout               : -1.5 Phase 1, episode ends
"""

from __future__ import annotations

# ── Phase 1 constants ──────────────────────────────────────────────────────
_WRONG_APPROVE   = -2.00
_WRONG_REJECT    = -1.00
_P1_TIMEOUT      = -1.50

# Correct decision scaling: reward = base + slope × info_confidence
_CORRECT_APPROVE_BASE  = -0.30   # blind correct approve — still negative
_CORRECT_APPROVE_SLOPE =  1.30   # × conf adds up to +1.00 at conf=1.0
_CORRECT_REJECT_BASE   =  0.25   # blind correct reject — conservative, small positive
_CORRECT_REJECT_SLOPE  =  0.45   # × conf adds up to +0.70 at conf=1.0

# Phase 1 step costs
DOC_REQUEST_COST = -0.10
FLAG_REVIEW_COST = -0.15

# ── Phase 2 constants ──────────────────────────────────────────────────────
REPAID_BONUS        =  1.50
DEFAULT_PENALTY     = -2.50
REMINDER_COST       = -0.05
RESTRUCTURE_COST    = -0.20
ESCALATE_COST       = -0.50


def info_confidence(income_revealed: bool, credit_history_revealed: bool) -> float:
    """
    Scalar in [0.0, 1.0] encoding how much evidence the agent has gathered.
    Credit history is weighted 2× income because it is the stronger signal.
    """
    return 0.35 * int(income_revealed) + 0.65 * int(credit_history_revealed)


def phase1_terminal_reward(
    decision: str,           # "APPROVE" | "REJECT"
    ground_truth: str,       # "APPROVE" | "REJECT"
    income_revealed: bool,
    credit_history_revealed: bool,
    step_penalties: float,   # cumulative doc/review costs (negative)
) -> float:
    """
    Compute the Phase 1 terminal reward.

    Key property: blind-approve is ALWAYS negative in expectation.
    The agent must gather evidence to justify an APPROVE decision.
    """
    conf    = info_confidence(income_revealed, credit_history_revealed)
    correct = (decision == ground_truth)

    if decision == "APPROVE":
        base = (
            _CORRECT_APPROVE_BASE + _CORRECT_APPROVE_SLOPE * conf
            if correct else _WRONG_APPROVE
        )
    else:  # REJECT
        base = (
            _CORRECT_REJECT_BASE + _CORRECT_REJECT_SLOPE * conf
            if correct else _WRONG_REJECT
        )

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

    REPAID : Phase 1 reward + bonus + intervention costs
    DEFAULT: Phase 1 reward replaced by harsh penalty
    ESCALATED: Phase 1 reward + escalate cost (cuts loss)
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
"""
models.py — Typed Pydantic contracts for the Microfinance Credit Decision Environment.

Action space  : Approve | Reject | RequestIncomeProof | RequestCreditHistory | FlagForReview
Observation   : Partial applicant profile + step cost + done signal
State         : Full episode metadata (hidden ground-truth lives here)
"""

from __future__ import annotations
from typing import Optional, List, Literal
from dataclasses import dataclass, field
from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

ActionType = Literal[
    "APPROVE",
    "REJECT",
    "REQUEST_INCOME_PROOF",
    "REQUEST_CREDIT_HISTORY",
    "FLAG_FOR_REVIEW",
]


@dataclass
class CreditAction(Action):
    """One decision step taken by the agent."""
    action_type: ActionType = "FLAG_FOR_REVIEW"
    # Optional free-text rationale (used by LLM grader only, no effect on env logic)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Observation  (what the agent can SEE — may contain None/null fields)
# ---------------------------------------------------------------------------

@dataclass
class ApplicantObservation(Observation):
    """
    Partial view of the applicant.  Fields are None until the relevant
    document-request action is taken.  This is intentional: the agent
    must decide *when* to pay the step cost to reveal hidden information.
    """
    # ── Always visible on reset ──────────────────────────────────────────
    applicant_id: str = ""
    dependents: Optional[int] = None          # always revealed at start
    occupation: Optional[str] = None          # always revealed at start
    loan_amount_requested: Optional[float] = None  # always revealed
    region_tier: Optional[str] = None         # rural / semi-urban / urban

    # ── Revealed only after REQUEST_INCOME_PROOF ─────────────────────────
    monthly_income: Optional[float] = None    # None until doc requested
    income_source_stability: Optional[str] = None  # "stable" | "variable" | "seasonal"

    # ── Revealed only after REQUEST_CREDIT_HISTORY ───────────────────────
    previous_loans: Optional[int] = None      # count
    past_defaults: Optional[int] = None       # count
    repayment_streak: Optional[int] = None    # consecutive on-time months

    # ── Revealed only after FLAG_FOR_REVIEW ──────────────────────────────
    # A senior-officer comment string (simulated) surfaces after flag
    senior_review_comment: Optional[str] = None

    # ── Step metadata ─────────────────────────────────────────────────────
    step_count: int = 0
    max_steps: int = 7
    documents_submitted: List[str] = field(default_factory=list)
    last_action_result: str = ""   # human-readable feedback for last action

    # ── Terminal ──────────────────────────────────────────────────────────
    # done + reward inherited from Observation base class


# ---------------------------------------------------------------------------
# State  (full episode metadata — includes hidden ground truth)
# ---------------------------------------------------------------------------

@dataclass
class MicrofinanceState(State):
    """
    Full episode state. Ground truth is stored here and never sent to the
    agent directly.  The server uses it to compute rewards.
    """
    # Ground truth (hidden from agent)
    ground_truth_label: Optional[str] = None   # "APPROVE" | "REJECT"
    true_default_probability: Optional[float] = None  # 0.0–1.0

    # Revealed field tracking
    income_revealed: bool = False
    credit_history_revealed: bool = False
    senior_review_done: bool = False

    # Full hidden profile (stored for grader)
    hidden_monthly_income: Optional[float] = None
    hidden_past_defaults: Optional[int] = None
    hidden_repayment_streak: Optional[int] = None
    hidden_income_stability: Optional[str] = None

    # Reward accounting
    cumulative_step_penalty: float = 0.0
    terminal_reward: Optional[float] = None
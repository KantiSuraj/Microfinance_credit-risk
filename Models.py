"""
models.py — Typed Pydantic contracts for the Microfinance Credit Decision Environment.

Action space  : Approve | Reject | RequestIncomeProof | RequestCreditHistory | FlagForReview
Observation   : Partial applicant profile + step cost + done signal
State         : Full episode metadata (hidden ground-truth lives here)

Key additions vs v1:
  - Phase enum         : APPLICATION | MONITORING | TERMINAL
  - PaymentStatus enum : ON_TIME | LATE | MISSED
  - MonitoringObservation: Phase 2 observation type
  - MicrofinanceState  : extended with all Phase 2 tracking fields

NOTE: Action, Observation, State from openenv are Pydantic BaseModels.
      Do NOT use @dataclass on subclasses — use plain Pydantic field syntax.
"""

from __future__ import annotations
from typing import Optional, List, Literal
from enum import Enum
from pydantic import Field
from openenv.core.env_server import Action, Observation, State
 
 
# ── Enums ──────────────────────────────────────────────────────────────────
 
class Phase(Enum):
    APPLICATION = "APPLICATION"
    MONITORING  = "MONITORING"
    TERMINAL    = "TERMINAL"
 
 
class PaymentStatus(Enum):
    ON_TIME = "ON_TIME"
    LATE    = "LATE"
    MISSED  = "MISSED"
 
 
# ── Action (shared across both phases) ────────────────────────────────────
# Phase 1 valid: APPROVE | REJECT | REQUEST_INCOME_PROOF |
#                REQUEST_CREDIT_HISTORY | FLAG_FOR_REVIEW
# Phase 2 valid: DO_NOTHING | SEND_REMINDER |
#                RESTRUCTURE_LOAN | ESCALATE_TO_RECOVERY
 
ActionType = Literal[
    # Phase 1
    "APPROVE", "REJECT",
    "REQUEST_INCOME_PROOF", "REQUEST_CREDIT_HISTORY", "FLAG_FOR_REVIEW",
    # Phase 2
    "DO_NOTHING", "SEND_REMINDER", "RESTRUCTURE_LOAN", "ESCALATE_TO_RECOVERY",
]
 
 
class CreditAction(Action):
    action_type: ActionType = "DO_NOTHING"
    rationale: str = ""
 
 
# ── Phase 1 Observation ────────────────────────────────────────────────────
 
class ApplicantObservation(Observation):
    """Partial applicant profile. Fields are None until revealed by doc requests."""
    # Always visible
    applicant_id           : str            = ""
    dependents             : Optional[int]  = None
    occupation             : Optional[str]  = None
    loan_amount_requested  : Optional[float]= None
    region_tier            : Optional[str]  = None
    # Revealed by REQUEST_INCOME_PROOF
    monthly_income         : Optional[float]= None
    income_source_stability: Optional[str]  = None
    # Revealed by REQUEST_CREDIT_HISTORY
    previous_loans         : Optional[int]  = None
    past_defaults          : Optional[int]  = None
    repayment_streak       : Optional[int]  = None
    # Revealed by FLAG_FOR_REVIEW
    senior_review_comment  : Optional[str]  = None
    # Step metadata
    step_count             : int            = 0
    max_steps              : int            = 7
    documents_submitted    : List[str]      = Field(default_factory=list)
    info_confidence        : float          = 0.0   # [0,1] — shown to agent
    current_phase          : str            = "APPLICATION"
    last_action_result     : str            = ""
    transitioning_to_phase2: bool           = False
 
 
# ── Phase 2 Observation ────────────────────────────────────────────────────
 
class MonitoringObservation(Observation):
    """
    Monthly repayment monitoring observation.
    The agent sees the *observed* payment (which may be noisy).
    The true underlying default probability is hidden.
    """
    month_number        : int         = 0
    months_remaining    : int         = 12
    # Noisy payment observation — may not match ground truth
    observed_payment    : str         = "ON_TIME"
    # Signal quality is revealed to the agent so it knows how much to trust obs
    signal_quality      : float       = 0.90
    # Hidden from agent in production (exposed here for grader / evaluation)
    current_default_prob_hidden: float = 0.0
    # Visible counters
    cumulative_misses   : int         = 0
    missed_streak       : int         = 0
    ontime_streak       : int         = 0
    payment_history     : List[str]   = Field(default_factory=list)
    intervention_history: List[str]   = Field(default_factory=list)
    # Noisy precursor signal [0,1]. Values above 0.5 suggest elevated
    # external shock probability next month. Zero when no shock is scheduled.
    economic_stress_signal: float     = 0.0
    current_phase       : str         = "MONITORING"
    last_action_result  : str         = ""
 
 
# ── Full Episode State (includes hidden ground truth) ─────────────────────
 
class MicrofinanceState(State):
    """
    Complete episode state. Ground truth fields are never sent to the agent
    directly — they live here for the server's reward computation and grader.
    """
    # ── Episode tracking ───────────────────────────────────────────────────
    episode_id           : Optional[str] = None
    step_count           : int         = 0
 
    # ── Phase tracking ─────────────────────────────────────────────────────
    phase                : Phase      = Phase.APPLICATION
 
    # ── Phase 1 hidden ground truth ────────────────────────────────────────
    ground_truth_label         : Optional[str]   = None   # "APPROVE" | "REJECT"
    true_default_probability   : Optional[float] = None
    hidden_monthly_income      : Optional[float] = None
    hidden_past_defaults       : Optional[int]   = None
    hidden_repayment_streak    : Optional[int]   = None
    hidden_income_stability    : Optional[str]   = None
 
    # ── Phase 1 progress ───────────────────────────────────────────────────
    income_revealed            : bool   = False
    credit_history_revealed    : bool   = False
    senior_review_done         : bool   = False
    cumulative_step_penalty    : float  = 0.0
    phase1_reward              : Optional[float] = None
 
    # ── Anti-hack tracking ────────────────────────────────────────────────
    redundant_actions          : int    = 0     # count of wasted/redundant actions
    docs_requested_count       : int    = 0     # total doc+review requests made
    consecutive_reminders      : int    = 0     # Phase 2 reminder spam counter
    # v2: Phase 2 action diversity tracking
    consecutive_same_action    : int    = 0     # streak of identical Phase 2 actions
    last_phase2_action         : Optional[str] = None  # last Phase 2 action taken
 
    # ── Phase 2 state ──────────────────────────────────────────────────────
    # Signal quality: set at Phase 1 → 2 boundary based on docs collected
    signal_quality             : Optional[float] = None
    # Evolving default probability (modified by interventions + payment streaks)
    current_default_prob       : float  = 0.5
    months_completed           : int    = 0
    total_months               : int    = 12
    payment_history            : List[str] = Field(default_factory=list)
    intervention_history       : List[str] = Field(default_factory=list)
    missed_streak              : int    = 0
    ontime_streak              : int    = 0
    cumulative_misses          : int    = 0
    phase2_intervention_costs  : float  = 0.0

    # ── External shock scheduling (1-month lagged signal) ─────────────────
    shock_scheduled            : bool   = False
    shock_magnitude            : float  = 0.0
    shock_signal_strength      : float  = 0.0
 
    # ── Terminal ───────────────────────────────────────────────────────────
    terminal_reward            : Optional[float] = None
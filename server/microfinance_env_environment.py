"""
microfinance_environment.py  —  Two-phase loan lifecycle RL environment.

═══════════════════════════════════════════════════════════════════════════
WHY THIS IS NOT A CLASSIFICATION PROBLEM
═══════════════════════════════════════════════════════════════════════════
A classifier sees a static feature vector and outputs APPROVE/REJECT.
This environment forces the agent to:

1. Plan across two phases with different action spaces.
2. Trade off Phase 1 information cost against Phase 2 signal quality:
   - Skipping credit history in Phase 1 means noisy payment observations
     in Phase 2 (60% accurate vs 90% if credit history was collected).
3. Make time-sensitive interventions: early action on a flagging borrower
   compounds over remaining months; late action is weaker.
4. Manage a portfolio-level tradeoff: tight approval is safe but misses
   creditworthy borrowers (financial inclusion cost).

EPISODE STRUCTURE
─────────────────
Phase 1  APPLICATION  (steps 1–7)
  Reset → partial obs → doc requests → APPROVE or REJECT
  If REJECT → terminal, reward is Phase 1 only.
  If APPROVE → Phase 2 begins.

Phase 2  MONITORING   (months 1–12, one step per month)
  Each step: agent observes payment signal (with noise), then acts.
  Actions: DO_NOTHING | SEND_REMINDER | RESTRUCTURE_LOAN | ESCALATE_TO_RECOVERY
  Terminal triggers:
    • All months complete without default → REPAID
    • Accumulated missed payments cross threshold → DEFAULT
    • ESCALATE_TO_RECOVERY called → forced write-off

SIGNAL QUALITY PROPAGATION
───────────────────────────
Phase 2 payment observations are Bernoulli draws:
  true_payment = (random() > borrower.default_prob_per_month)
  observed_payment = (
      true_payment  with probability signal_quality,
      ~true_payment with probability (1 - signal_quality)
  )

signal_quality is set at Phase 1 → Phase 2 transition:
  credit_history collected → 0.90
  income_proof only         → 0.75
  neither                   → 0.60

This means an agent that blindly approves without gathering docs will see
noisy month-by-month signals and struggle to identify at-risk borrowers
before it is too late to intervene.

INTERVENTION DYNAMICS
──────────────────────
Each intervention modifies the borrower's per-month default probability:
  DO_NOTHING          : no change (default risk accumulates via missed_streak)
  SEND_REMINDER       : default_prob *= 0.95  (−5%, cost 0.05)
  RESTRUCTURE_LOAN    : default_prob *= 0.80  (−20%, cost 0.20)
  ESCALATE_TO_RECOVERY: episode ends immediately, moderate penalty (−0.5)

Consecutive missed payments increase default_prob by +0.04/month (stress).
Consecutive on-time payments reduce it by −0.02/month (trust building).
"""

from __future__ import annotations
import random
import uuid
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field
from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
    CreditAction, ApplicantObservation,
    MonitoringObservation, MicrofinanceState,
    Phase, PaymentStatus,
)
from server.data_generator import generate_dataset, ApplicantProfile, SENIOR_COMMENTS

# ── Reward constants ───────────────────────────────────────────────────────
R_CORRECT_APPROVE    =  1.0
R_WRONG_APPROVE      = -2.0
R_CORRECT_REJECT     =  0.6
R_WRONG_REJECT       = -1.0
R_DOC_REQUEST        = -0.10
R_FLAG_REVIEW        = -0.15
R_MAX_STEPS_PHASE1   = -1.5

R_LOAN_REPAID        =  1.5   # bonus added to approval reward
R_LOAN_DEFAULT       = -2.5   # replaces approval reward entirely
R_SEND_REMINDER      = -0.05
R_RESTRUCTURE        = -0.20
R_ESCALATE_RECOVERY  = -0.50

# ── Phase 2 dynamics ───────────────────────────────────────────────────────
MISSED_STREAK_PENALTY    = +0.04   # default_prob increase per consecutive miss
ONTIME_STREAK_BONUS      = -0.02   # default_prob reduction per consecutive on-time
DEFAULT_THRESHOLD_MISSES = 4       # cumulative misses that trigger auto-default
MAX_MONITORING_MONTHS    = 12
MAX_PHASE1_STEPS         = 7

# ── Signal quality mapping ─────────────────────────────────────────────────
def _signal_quality(income_revealed: bool, credit_revealed: bool) -> float:
    if credit_revealed:
        return 0.90
    elif income_revealed:
        return 0.75
    else:
        return 0.60


class MicrofinanceEnvironment(Environment):
    """
    Microfinance Credit Decision Environment — two-phase, long-horizon.

    Action space changes between phases:
      Phase 1: APPROVE | REJECT | REQUEST_INCOME_PROOF |
               REQUEST_CREDIT_HISTORY | FLAG_FOR_REVIEW
      Phase 2: DO_NOTHING | SEND_REMINDER |
               RESTRUCTURE_LOAN | ESCALATE_TO_RECOVERY
    """

    def __init__(self, dataset_size: int = 300, seed: int = 42):
        super().__init__()
        self._dataset = generate_dataset(n=dataset_size, seed=seed)
        self._rng     = random.Random(seed)
        self._profile : Optional[ApplicantProfile]  = None
        self._state   : Optional[MicrofinanceState] = None

    # ══════════════════════════════════════════════════════════════════════
    # OpenEnv interface
    # ══════════════════════════════════════════════════════════════════════

    def reset(self):
        profile = self._rng.choice(self._dataset)
        self._profile = profile

        self._state = MicrofinanceState(
            episode_id           = str(uuid.uuid4()),
            step_count           = 0,
            phase                = Phase.APPLICATION,
            # Phase 1 tracking
            ground_truth_label   = profile.ground_truth_label,
            true_default_probability = profile.true_default_probability,
            income_revealed      = False,
            credit_history_revealed = False,
            senior_review_done   = False,
            hidden_monthly_income = profile.monthly_income,
            hidden_past_defaults  = profile.past_defaults,
            hidden_repayment_streak = profile.repayment_streak,
            hidden_income_stability = profile.income_source_stability,
            cumulative_step_penalty = 0.0,
            phase1_reward        = None,
            # Phase 2 tracking
            signal_quality       = None,   # set at Phase 1 → 2 transition
            current_default_prob = profile.true_default_probability,
            months_completed     = 0,
            total_months         = MAX_MONITORING_MONTHS,
            payment_history      = [],     # list of "ON_TIME"/"LATE"/"MISSED"
            intervention_history = [],
            missed_streak        = 0,
            ontime_streak        = 0,
            cumulative_misses    = 0,
            terminal_reward      = None,
        )
        return self._build_phase1_obs("Application received. Income and credit history pending.")

    # ──────────────────────────────────────────────────────────────────────

    def step(self, action: CreditAction):
        s = self._state
        if s is None:
            raise RuntimeError("Call reset() before step().")

        s.step_count += 1

        if s.phase == Phase.APPLICATION:
            return self._phase1_step(action)
        else:
            return self._phase2_step(action)

    @property
    def state(self) -> MicrofinanceState:
        if self._state is None:
            raise RuntimeError("Not initialised.")
        return self._state

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1 logic
    # ══════════════════════════════════════════════════════════════════════

    def _phase1_step(self, action: CreditAction):
        s   = self._state
        p   = self._profile
        atype = action.action_type

        # ── Max steps guard ───────────────────────────────────────────────
        if s.step_count >= MAX_PHASE1_STEPS and atype not in ("APPROVE", "REJECT"):
            reward = R_MAX_STEPS_PHASE1 + s.cumulative_step_penalty
            s.terminal_reward = reward
            s.phase = Phase.TERMINAL
            return self._build_phase1_obs(
                f"⚠ Phase 1 timeout. Auto-rejected. "
                f"Ground truth: {p.ground_truth_label}. Reward: {reward}.",
                done=True, reward=reward,
            )

        if atype == "REQUEST_INCOME_PROOF":
            if s.income_revealed:
                msg = "Income proof already on file."
            else:
                s.income_revealed = True
                s.cumulative_step_penalty += R_DOC_REQUEST
                msg = (f"Income proof received. "
                       f"₹{p.monthly_income:,.0f}/month, {p.income_source_stability} income. "
                       f"Step cost: {R_DOC_REQUEST}.")

        elif atype == "REQUEST_CREDIT_HISTORY":
            if s.credit_history_revealed:
                msg = "Credit history already on file."
            else:
                s.credit_history_revealed = True
                s.cumulative_step_penalty += R_DOC_REQUEST
                msg = (f"Credit history received. "
                       f"{p.previous_loans} prior loans, {p.past_defaults} defaults, "
                       f"{p.repayment_streak}-month streak. "
                       f"Step cost: {R_DOC_REQUEST}.")

        elif atype == "FLAG_FOR_REVIEW":
            if s.senior_review_done:
                msg = "Already escalated."
            else:
                s.senior_review_done = True
                s.cumulative_step_penalty += R_FLAG_REVIEW
                comment = self._rng.choice(SENIOR_COMMENTS[p.risk_band])
                msg = f"Senior review: '{comment}'. Step cost: {R_FLAG_REVIEW}."

        elif atype == "REJECT":
            correct = (p.ground_truth_label == "REJECT")
            base    = R_CORRECT_REJECT if correct else R_WRONG_REJECT
            reward  = base + s.cumulative_step_penalty
            s.terminal_reward = reward
            s.phase1_reward   = reward
            s.phase = Phase.TERMINAL
            return self._build_phase1_obs(
                f"REJECTED. {'✓ Correct' if correct else '✗ Incorrect'}. "
                f"Ground truth: {p.ground_truth_label}. "
                f"Default prob: {p.true_default_probability:.2f}. "
                f"Episode reward: {reward:.3f}.",
                done=True, reward=reward,
            )

        elif atype == "APPROVE":
            correct   = (p.ground_truth_label == "APPROVE")
            base      = R_CORRECT_APPROVE if correct else R_WRONG_APPROVE
            p1_reward = base + s.cumulative_step_penalty
            s.phase1_reward = p1_reward

            # ── Transition to Phase 2 ──────────────────────────────────
            sq = _signal_quality(s.income_revealed, s.credit_history_revealed)
            s.signal_quality       = sq
            s.current_default_prob = p.true_default_probability
            s.phase                = Phase.MONITORING
            s.months_completed     = 0

            return self._build_phase1_obs(
                f"APPROVED. Phase 2 begins. "
                f"Signal quality: {sq:.0%} "
                f"({'credit history collected' if s.credit_history_revealed else 'no credit history — noisy observations'}). "
                f"Ground truth: {p.ground_truth_label}.",
                done=False, reward=p1_reward,
            )

        else:
            msg = f"Unknown action '{atype}' in Phase 1."

        return self._build_phase1_obs(msg, done=False, reward=s.cumulative_step_penalty)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2 logic
    # ══════════════════════════════════════════════════════════════════════

    def _phase2_step(self, action: CreditAction):
        s   = self._state
        p   = self._profile
        atype = action.action_type

        s.months_completed += 1
        month = s.months_completed

        # ── Simulate this month's payment ─────────────────────────────────
        true_paid = self._rng.random() > s.current_default_prob

        # Noisy observation: flip true_paid with prob (1 - signal_quality)
        if self._rng.random() < s.signal_quality:
            observed_paid = true_paid
        else:
            observed_paid = not true_paid   # agent sees the wrong signal

        # Payment status label for observation
        if true_paid:
            true_status = PaymentStatus.ON_TIME
        else:
            true_status = PaymentStatus.MISSED

        observed_status = PaymentStatus.ON_TIME if observed_paid else PaymentStatus.MISSED

        # Update streak counters from TRUE payment
        if true_paid:
            s.missed_streak   = 0
            s.ontime_streak  += 1
            s.current_default_prob = max(
                0.02,
                s.current_default_prob + ONTIME_STREAK_BONUS * min(s.ontime_streak, 3)
            )
        else:
            s.ontime_streak  = 0
            s.missed_streak += 1
            s.cumulative_misses += 1
            s.current_default_prob = min(
                0.98,
                s.current_default_prob + MISSED_STREAK_PENALTY * s.missed_streak
            )

        s.payment_history.append(true_status.value)

        # ── Process agent's intervention ──────────────────────────────────
        intervention_cost = 0.0
        intervention_note = ""

        if atype == "SEND_REMINDER":
            s.current_default_prob = max(0.02, s.current_default_prob * 0.95)
            intervention_cost = R_SEND_REMINDER
            intervention_note = f"Reminder sent. Default risk reduced to {s.current_default_prob:.2%}."
            s.intervention_history.append(f"M{month}:REMINDER")

        elif atype == "RESTRUCTURE_LOAN":
            s.current_default_prob = max(0.02, s.current_default_prob * 0.80)
            intervention_cost = R_RESTRUCTURE
            intervention_note = f"Loan restructured. Default risk reduced to {s.current_default_prob:.2%}."
            s.intervention_history.append(f"M{month}:RESTRUCTURE")

        elif atype == "ESCALATE_TO_RECOVERY":
            intervention_cost = R_ESCALATE_RECOVERY
            final_reward = (s.phase1_reward or 0.0) + R_ESCALATE_RECOVERY
            s.terminal_reward = final_reward
            s.phase = Phase.TERMINAL
            return self._build_monitoring_obs(
                month, observed_status,
                f"Escalated to recovery. Episode ends. Reward: {final_reward:.3f}.",
                done=True, reward=final_reward,
                intervention_note=intervention_note,
            )

        # ── Check terminal conditions ─────────────────────────────────────

        # Auto-default on too many cumulative misses
        if s.cumulative_misses >= DEFAULT_THRESHOLD_MISSES:
            final_reward = (s.phase1_reward or 0.0) + R_LOAN_DEFAULT
            s.terminal_reward = final_reward
            s.phase = Phase.TERMINAL
            return self._build_monitoring_obs(
                month, observed_status,
                f"DEFAULT triggered after {s.cumulative_misses} missed payments. "
                f"True default prob at termination: {s.current_default_prob:.2%}. "
                f"Episode reward: {final_reward:.3f}.",
                done=True, reward=final_reward,
                intervention_note=intervention_note,
            )

        # Successfully completed all months
        if s.months_completed >= MAX_MONITORING_MONTHS:
            final_reward = (s.phase1_reward or 0.0) + R_LOAN_REPAID + intervention_cost
            s.terminal_reward = final_reward
            s.phase = Phase.TERMINAL
            return self._build_monitoring_obs(
                month, observed_status,
                f"Loan fully repaid after {MAX_MONITORING_MONTHS} months. "
                f"Episode reward: {final_reward:.3f}.",
                done=True, reward=final_reward,
                intervention_note=intervention_note,
            )

        # Continue episode
        months_left = MAX_MONITORING_MONTHS - s.months_completed
        return self._build_monitoring_obs(
            month, observed_status,
            f"Month {month}/{MAX_MONITORING_MONTHS}. "
            f"Observed: {observed_status.value}. "
            f"Cumulative misses (true): {s.cumulative_misses}. "
            f"{months_left} months remaining. "
            f"{intervention_note}",
            done=False,
            reward=(s.phase1_reward or 0.0) + intervention_cost,
            intervention_note=intervention_note,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Observation builders
    # ══════════════════════════════════════════════════════════════════════

    def _build_phase1_obs(
        self, msg: str, *, done: bool = False, reward: float = 0.0
    ) -> ApplicantObservation:
        s = self._state
        p = self._profile
        return ApplicantObservation(
            applicant_id           = p.applicant_id,
            dependents             = p.dependents,
            occupation             = p.occupation,
            loan_amount_requested  = p.loan_amount_requested,
            region_tier            = p.region_tier,
            monthly_income         = p.monthly_income if s.income_revealed else None,
            income_source_stability= p.income_source_stability if s.income_revealed else None,
            previous_loans         = p.previous_loans if s.credit_history_revealed else None,
            past_defaults          = p.past_defaults if s.credit_history_revealed else None,
            repayment_streak       = p.repayment_streak if s.credit_history_revealed else None,
            senior_review_comment  = None,
            step_count             = s.step_count,
            max_steps              = MAX_PHASE1_STEPS,
            documents_submitted    = (
                (["income_proof"] if s.income_revealed else []) +
                (["credit_history"] if s.credit_history_revealed else []) +
                (["senior_review"] if s.senior_review_done else [])
            ),
            current_phase          = s.phase.value,
            last_action_result     = msg,
            done                   = done,
            reward                 = reward,
        )

    def _build_monitoring_obs(
        self, month: int, observed_status,
        msg: str, *, done: bool, reward: float,
        intervention_note: str = "",
    ) -> MonitoringObservation:
        s = self._state
        return MonitoringObservation(
            month_number          = month,
            months_remaining      = MAX_MONITORING_MONTHS - month,
            observed_payment      = observed_status.value,
            signal_quality        = s.signal_quality,
            current_default_prob_hidden = s.current_default_prob,  # hidden in real use
            cumulative_misses     = s.cumulative_misses,
            payment_history       = list(s.payment_history),
            intervention_history  = list(s.intervention_history),
            missed_streak         = s.missed_streak,
            ontime_streak         = s.ontime_streak,
            current_phase         = s.phase.value,
            last_action_result    = msg,
            done                  = done,
            reward                = reward,
        )
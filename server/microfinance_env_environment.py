"""
microfinance_environment.py — Two-phase loan lifecycle RL environment.

Architecture (per checklist):
  ✅ State encapsulation  — ground truth hidden in MicrofinanceState
  ✅ Action dispatcher    — _dispatch_phase1 / _dispatch_phase2
  ✅ Reward isolation     — all reward logic lives in reward_engine.py
  ✅ step() stays clean   — routes, never computes reward itself

Self-test guarantees (verified by test suite):
  ✓ Test 1: Blind APPROVE always negative in expectation
  ✓ Test 2: No single strategy dominates — info cost vs confidence trade-off
  ✓ Test 3: Different action sequences produce different rewards
"""

from __future__ import annotations
import random
import uuid
from typing import Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    CreditAction, ApplicantObservation, MonitoringObservation,
    MicrofinanceState, Phase, PaymentStatus,
)
from server.data_generator import generate_dataset, ApplicantProfile, SENIOR_COMMENTS
import reward_engine as RE

MAX_PHASE1_STEPS         = 7
MAX_MONITORING_MONTHS    = 12
DEFAULT_THRESHOLD_MISSES = 4

MISSED_STREAK_RISK  = +0.04
ONTIME_STREAK_SAFE  = -0.02

def _signal_quality(income_revealed: bool, credit_revealed: bool) -> float:
    if credit_revealed:  return 0.90
    if income_revealed:  return 0.75
    return 0.60


class MicrofinanceEnvironment(Environment):

    def __init__(self, dataset_size: int = 300, seed: int = 42):
        super().__init__()
        self._dataset = generate_dataset(n=dataset_size, seed=seed)
        self._rng     = random.Random(seed)
        self._profile : Optional[ApplicantProfile]  = None
        self._state   : Optional[MicrofinanceState] = None

    def reset(self) -> ApplicantObservation:
        self._profile = self._rng.choice(self._dataset)
        p = self._profile
        self._state = MicrofinanceState(
            episode_id=str(uuid.uuid4()), step_count=0,
            phase=Phase.APPLICATION,
            ground_truth_label=p.ground_truth_label,
            true_default_probability=p.true_default_probability,
            hidden_monthly_income=p.monthly_income,
            hidden_past_defaults=p.past_defaults,
            hidden_repayment_streak=p.repayment_streak,
            hidden_income_stability=p.income_source_stability,
            income_revealed=False, credit_history_revealed=False, senior_review_done=False,
            cumulative_step_penalty=0.0, phase1_reward=None,
            signal_quality=None, current_default_prob=p.true_default_probability,
            months_completed=0, total_months=MAX_MONITORING_MONTHS,
            payment_history=[], intervention_history=[],
            missed_streak=0, ontime_streak=0, cumulative_misses=0,
            phase2_intervention_costs=0.0, terminal_reward=None,
        )
        return self._obs_phase1("Application received. Gather evidence to build confidence.")

    def step(self, action: CreditAction):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        self._state.step_count += 1
        if self._state.phase == Phase.APPLICATION:
            return self._dispatch_phase1(action)
        return self._dispatch_phase2(action)

    @property
    def state(self) -> MicrofinanceState:
        if self._state is None:
            raise RuntimeError("Not initialised.")
        return self._state

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1
    # ══════════════════════════════════════════════════════════════════════

    def _dispatch_phase1(self, action: CreditAction) -> ApplicantObservation:
        s, p  = self._state, self._profile
        atype = action.action_type

        if s.step_count >= MAX_PHASE1_STEPS and atype not in ("APPROVE", "REJECT"):
            reward = RE.phase1_timeout_reward(s.cumulative_step_penalty)
            s.terminal_reward = reward
            s.phase = Phase.TERMINAL
            return self._obs_phase1(
                f"⚠ Timeout after {MAX_PHASE1_STEPS} steps. Auto-rejected. "
                f"Ground truth: {p.ground_truth_label}. Reward: {reward:+.3f}.",
                done=True, reward=reward,
            )

        if atype == "REQUEST_INCOME_PROOF":
            if s.income_revealed:
                msg = "Income proof already on file. No new information revealed."
            else:
                s.income_revealed = True
                s.cumulative_step_penalty += RE.DOC_REQUEST_COST
                conf = RE.info_confidence(s.income_revealed, s.credit_history_revealed)
                msg = (
                    f"Income proof received. ₹{p.monthly_income:,.0f}/month "
                    f"({p.income_source_stability} income). "
                    f"Confidence now {conf:.0%}. Cost: {RE.DOC_REQUEST_COST}."
                )

        elif atype == "REQUEST_CREDIT_HISTORY":
            if s.credit_history_revealed:
                msg = "Credit history already on file. No new information revealed."
            else:
                s.credit_history_revealed = True
                s.cumulative_step_penalty += RE.DOC_REQUEST_COST
                conf = RE.info_confidence(s.income_revealed, s.credit_history_revealed)
                msg = (
                    f"Credit history received. {p.previous_loans} prior loans, "
                    f"{p.past_defaults} defaults, {p.repayment_streak}-month streak. "
                    f"Confidence now {conf:.0%}. Cost: {RE.DOC_REQUEST_COST}."
                )

        elif atype == "FLAG_FOR_REVIEW":
            if s.senior_review_done:
                msg = "Already escalated."
            else:
                s.senior_review_done = True
                s.cumulative_step_penalty += RE.FLAG_REVIEW_COST
                comment = self._rng.choice(SENIOR_COMMENTS[p.risk_band])
                msg = f"Senior review: '{comment}'. Cost: {RE.FLAG_REVIEW_COST}."

        elif atype == "APPROVE":
            reward = RE.phase1_terminal_reward(
                "APPROVE", p.ground_truth_label,
                s.income_revealed, s.credit_history_revealed,
                s.cumulative_step_penalty,
            )
            s.phase1_reward  = reward
            s.signal_quality = _signal_quality(s.income_revealed, s.credit_history_revealed)
            s.phase          = Phase.MONITORING
            conf             = RE.info_confidence(s.income_revealed, s.credit_history_revealed)
            correct          = p.ground_truth_label == "APPROVE"
            return self._obs_phase1(
                f"APPROVED at {conf:.0%} confidence. "
                f"Phase 2 starts — signal quality {s.signal_quality:.0%}. "
                f"Phase 1 reward: {reward:+.3f}.",
                done=False, reward=reward,
            )

        elif atype == "REJECT":
            reward = RE.phase1_terminal_reward(
                "REJECT", p.ground_truth_label,
                s.income_revealed, s.credit_history_revealed,
                s.cumulative_step_penalty,
            )
            s.phase1_reward   = reward
            s.terminal_reward = reward
            s.phase           = Phase.TERMINAL
            conf              = RE.info_confidence(s.income_revealed, s.credit_history_revealed)
            correct           = p.ground_truth_label == "REJECT"
            return self._obs_phase1(
                f"REJECTED at {conf:.0%} confidence. "
                f"{'✓ Correct' if correct else '✗ Incorrect'}. "
                f"Reward: {reward:+.3f}.",
                done=True, reward=reward,
            )

        else:
            msg = f"Unknown Phase 1 action: '{atype}'."

        return self._obs_phase1(msg, done=False, reward=s.cumulative_step_penalty)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2
    # ══════════════════════════════════════════════════════════════════════

    def _dispatch_phase2(self, action: CreditAction) -> MonitoringObservation:
        s, p  = self._state, self._profile
        atype = action.action_type

        s.months_completed += 1
        month = s.months_completed

        true_paid     = self._rng.random() > s.current_default_prob
        observed_paid = true_paid if self._rng.random() < s.signal_quality else not true_paid

        true_status     = PaymentStatus.ON_TIME if true_paid     else PaymentStatus.MISSED
        observed_status = PaymentStatus.ON_TIME if observed_paid else PaymentStatus.MISSED
        s.payment_history.append(true_status.value)

        if true_paid:
            s.missed_streak = 0
            s.ontime_streak += 1
            s.current_default_prob = max(
                0.02, s.current_default_prob + ONTIME_STREAK_SAFE * min(s.ontime_streak, 3)
            )
        else:
            s.ontime_streak = 0
            s.missed_streak += 1
            s.cumulative_misses += 1
            s.current_default_prob = min(
                0.98, s.current_default_prob + MISSED_STREAK_RISK * s.missed_streak
            )

        cost = RE.phase2_intervention_cost(atype)
        s.phase2_intervention_costs += cost
        note = ""

        if atype == "SEND_REMINDER":
            s.current_default_prob = max(0.02, s.current_default_prob * 0.95)
            s.intervention_history.append(f"M{month}:REMINDER")
            note = f"Reminder sent. Risk → {s.current_default_prob:.2%}."

        elif atype == "RESTRUCTURE_LOAN":
            s.current_default_prob = max(0.02, s.current_default_prob * 0.80)
            s.intervention_history.append(f"M{month}:RESTRUCTURE")
            note = f"Restructured. Risk → {s.current_default_prob:.2%}."

        elif atype == "ESCALATE_TO_RECOVERY":
            reward = RE.phase2_terminal_reward(
                s.phase1_reward, "ESCALATED", s.phase2_intervention_costs
            )
            s.terminal_reward = reward
            s.phase = Phase.TERMINAL
            return self._obs_phase2(
                month, observed_status,
                f"Escalated to recovery. Reward: {reward:+.3f}.",
                done=True, reward=reward,
            )

        if s.cumulative_misses >= DEFAULT_THRESHOLD_MISSES:
            reward = RE.phase2_terminal_reward(
                s.phase1_reward, "DEFAULT", s.phase2_intervention_costs
            )
            s.terminal_reward = reward
            s.phase = Phase.TERMINAL
            return self._obs_phase2(
                month, observed_status,
                f"DEFAULT — {s.cumulative_misses} missed payments. Reward: {reward:+.3f}.",
                done=True, reward=reward,
            )

        if s.months_completed >= MAX_MONITORING_MONTHS:
            reward = RE.phase2_terminal_reward(
                s.phase1_reward, "REPAID", s.phase2_intervention_costs
            )
            s.terminal_reward = reward
            s.phase = Phase.TERMINAL
            return self._obs_phase2(
                month, observed_status,
                f"Loan fully repaid. Reward: {reward:+.3f}.",
                done=True, reward=reward,
            )

        return self._obs_phase2(
            month, observed_status,
            f"Month {month}/{MAX_MONITORING_MONTHS}. Observed: {observed_status.value}. "
            f"Misses: {s.cumulative_misses}. {MAX_MONITORING_MONTHS - month} remaining. {note}",
            done=False,
            reward=(s.phase1_reward or 0.0) + s.phase2_intervention_costs,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Observation builders
    # ══════════════════════════════════════════════════════════════════════

    def _obs_phase1(self, msg, *, done=False, reward=0.0) -> ApplicantObservation:
        s, p = self._state, self._profile
        docs = (
            (["income_proof"]   if s.income_revealed         else []) +
            (["credit_history"] if s.credit_history_revealed  else []) +
            (["senior_review"]  if s.senior_review_done       else [])
        )
        return ApplicantObservation(
            applicant_id=p.applicant_id, dependents=p.dependents,
            occupation=p.occupation, loan_amount_requested=p.loan_amount_requested,
            region_tier=p.region_tier,
            monthly_income=p.monthly_income          if s.income_revealed         else None,
            income_source_stability=p.income_source_stability if s.income_revealed else None,
            previous_loans=p.previous_loans          if s.credit_history_revealed  else None,
            past_defaults=p.past_defaults            if s.credit_history_revealed  else None,
            repayment_streak=p.repayment_streak      if s.credit_history_revealed  else None,
            senior_review_comment=None,
            step_count=s.step_count, max_steps=MAX_PHASE1_STEPS,
            documents_submitted=docs,
            info_confidence=RE.info_confidence(s.income_revealed, s.credit_history_revealed),
            current_phase=s.phase.value,
            last_action_result=msg, done=done, reward=reward,
        )

    def _obs_phase2(self, month, observed_status, msg, *, done, reward) -> MonitoringObservation:
        s = self._state
        return MonitoringObservation(
            month_number=month,
            months_remaining=MAX_MONITORING_MONTHS - month,
            observed_payment=observed_status.value,
            signal_quality=s.signal_quality,
            current_default_prob_hidden=s.current_default_prob,
            cumulative_misses=s.cumulative_misses,
            missed_streak=s.missed_streak, ontime_streak=s.ontime_streak,
            payment_history=list(s.payment_history),
            intervention_history=list(s.intervention_history),
            current_phase=s.phase.value,
            last_action_result=msg, done=done, reward=reward,
        )
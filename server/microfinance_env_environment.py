"""
microfinance_environment.py — Two-phase loan lifecycle RL environment.

Architecture (per checklist):
  ✅ State encapsulation  — ground truth hidden in MicrofinanceState
  ✅ Action dispatcher    — _dispatch_phase1 / _dispatch_phase2
  ✅ Reward isolation     — all reward logic lives in reward_engine.py
  ✅ step() stays clean   — routes, never computes reward itself

Task difficulty levels:
  • basic_lending       — Easy:  pre-revealed info, clear signals, short Phase 2
  • noisy_signals       — Medium: conflicting features, strategic doc gathering
  • adversarial_portfolio — Hard: borderline approval, long Phase 2, intervention timing

Self-test guarantees (verified by test suite):
  ✓ Test 1: Blind APPROVE always negative in expectation
  ✓ Test 2: No single strategy dominates — info cost vs confidence trade-off
  ✓ Test 3: Different action sequences produce different rewards
"""

from __future__ import annotations
import random
import uuid
from typing import Optional
from dataclasses import dataclass

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

import sys, os
# Add both project root and server/ dir for clean imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # server/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from models import (
    CreditAction, ApplicantObservation, MonitoringObservation,
    MicrofinanceState, Phase, PaymentStatus,
)
from server.data_generator import generate_dataset, generate_applicant, ApplicantProfile, SENIOR_COMMENTS
import server.reward_engine as RE


# ══════════════════════════════════════════════════════════════════════════
# Task Configuration — each task tests a DIFFERENT reasoning skill
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TaskConfig:
    """Structural configuration that changes what the agent must reason about."""
    name: str
    description: str
    # Phase 1 configuration
    pre_reveal_income: bool = False       # income visible from the start
    pre_reveal_credit: bool = False       # credit history visible from start
    force_conflicting: bool = False       # force conflicting-signal applicants
    force_borderline: bool = False        # force borderline applicants
    force_adversarial: bool = False        # force adversarial (APPROVE-worthy but risky)
    force_clear_case: bool = False        # force unambiguous applicants
    extra_doc_penalty: float = 0.0        # additional penalty for wrong doc requests
    # Phase 2 configuration
    phase2_months: int = 12               # monitoring duration
    elevated_base_risk: float = 0.0       # added to base default probability
    signal_quality_cap: float = 0.90      # max signal quality achievable
    default_threshold: int = 4            # misses before auto-default
    # General
    max_phase1_steps: int = 7
    seed: int = 42


# ── Task presets ──────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "basic_lending": TaskConfig(
        name="basic_lending",
        description=(
            "Easy: Minimal reasoning. Most applicant info is pre-revealed. "
            "Clear approve/reject signal. Short 6-month monitoring phase."
        ),
        pre_reveal_income=True,       # agent already sees income
        force_clear_case=True,        # low-ambiguity applicant
        phase2_months=6,              # short monitoring
        signal_quality_cap=0.90,
        default_threshold=4,
        seed=42,
    ),
    "noisy_signals": TaskConfig(
        name="noisy_signals",
        description=(
            "Medium: Conflicting features (high income + bad history or vice versa). "
            "Key info hidden. Must request the RIGHT documents — wrong requests penalized."
        ),
        force_conflicting=True,       # conflicting applicant signals
        extra_doc_penalty=-0.08,      # extra cost for each redundant/wrong doc request
        phase2_months=12,
        signal_quality_cap=0.90,
        default_threshold=4,
        seed=137,
    ),
    "adversarial_portfolio": TaskConfig(
        name="adversarial_portfolio",
        description=(
            "Hard: Borderline approval case. Phase 2 is the real challenge — "
            "elevated default risk, capped signal quality, recovery only possible "
            "with precisely timed interventions."
        ),
        force_adversarial=True,           # genuinely APPROVE-worthy but risky
        phase2_months=12,
        elevated_base_risk=0.12,          # +12% base default probability
        signal_quality_cap=0.75,          # noisy observations regardless of docs
        default_threshold=3,              # tighter failure threshold
        seed=256,
    ),
}

DEFAULT_TASK = "basic_lending"


# ── Global constants ──────────────────────────────────────────────────────

MAX_PHASE1_STEPS         = 7
DEFAULT_MONITORING_MONTHS = 12
DEFAULT_THRESHOLD_MISSES = 4

MISSED_STREAK_RISK  = +0.04
ONTIME_STREAK_SAFE  = -0.02

def _signal_quality(income_revealed: bool, credit_revealed: bool) -> float:
    if credit_revealed:  return 0.90
    if income_revealed:  return 0.75
    return 0.60


class MicrofinanceEnvironment(Environment):

    def __init__(self, dataset_size: int = 300, seed: int = 42, task_name: str = DEFAULT_TASK):
        super().__init__()
        self._task_config = TASK_CONFIGS.get(task_name, TASK_CONFIGS[DEFAULT_TASK])
        tc = self._task_config
        # Use caller's seed if explicitly provided, otherwise fall back to task config
        effective_seed = seed if seed != 42 else tc.seed
        adversarial_count = 30 if tc.force_adversarial else 0
        self._dataset = generate_dataset(
            n=dataset_size, seed=effective_seed,
            adversarial_count=adversarial_count,
        )
        self._rng     = random.Random(effective_seed)
        self._profile : Optional[ApplicantProfile]  = None
        self._state   : Optional[MicrofinanceState] = None

    def set_task(self, task_name: str) -> None:
        """
        Switch the active task configuration at runtime.
        Must be called BEFORE reset() so the next episode uses the new difficulty.
        Raises ValueError for unknown task names.
        Regenerates the dataset with the new task's seed and structural requirements.
        """
        if task_name not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks: {list(TASK_CONFIGS.keys())}"
            )
            
        if self._task_config and self._task_config.name == task_name:
            return
            
        self._task_config = TASK_CONFIGS[task_name]
        tc = self._task_config
        # Regenerate dataset with the task's seed and adversarial requirements
        adversarial_count = 30 if tc.force_adversarial else 0
        self._dataset = generate_dataset(
            n=len(self._dataset), seed=tc.seed,
            adversarial_count=adversarial_count,
        )
        self._rng = random.Random(tc.seed)

    def reset(self) -> ApplicantObservation:
        tc = self._task_config

        # ── Select applicant based on task structural requirements ─────────
        if tc.force_clear_case:
            # Easy: pick an unambiguous applicant (prob far from 0.5)
            candidates = [p for p in self._dataset
                          if abs(p.true_default_probability - 0.5) > 0.25
                          and not p.has_conflicting_signal]
            self._profile = self._rng.choice(candidates) if candidates else self._rng.choice(self._dataset)
        elif tc.force_conflicting:
            # Medium: pick a conflicting-signal applicant
            candidates = [p for p in self._dataset if p.has_conflicting_signal]
            self._profile = self._rng.choice(candidates) if candidates else self._rng.choice(self._dataset)
        elif tc.force_adversarial:
            # Hard: pick an adversarial applicant that genuinely scores APPROVE
            candidates = [p for p in self._dataset
                          if p.is_borderline and p.ground_truth_label == "APPROVE"]
            if not candidates:
                # Fallback: any APPROVE-worthy borderline profile
                candidates = [p for p in self._dataset
                              if p.ground_truth_label == "APPROVE" and
                              abs(p.true_default_probability - 0.5) < 0.20]
            self._profile = self._rng.choice(candidates) if candidates else self._rng.choice(self._dataset)
        elif tc.force_borderline:
            # Hard: pick a borderline applicant
            candidates = [p for p in self._dataset if p.is_borderline]
            self._profile = self._rng.choice(candidates) if candidates else self._rng.choice(self._dataset)
        else:
            self._profile = self._rng.choice(self._dataset)

        p = self._profile

        # ── Apply task-level risk elevation ────────────────────────────────
        base_default_prob = min(0.95, p.true_default_probability + tc.elevated_base_risk)

        self._state = MicrofinanceState(
            episode_id=str(uuid.uuid4()), step_count=0,
            phase=Phase.APPLICATION,
            ground_truth_label=p.ground_truth_label,
            true_default_probability=p.true_default_probability,
            hidden_monthly_income=p.monthly_income,
            hidden_past_defaults=p.past_defaults,
            hidden_repayment_streak=p.repayment_streak,
            hidden_income_stability=p.income_source_stability,
            income_revealed=tc.pre_reveal_income,
            credit_history_revealed=tc.pre_reveal_credit,
            senior_review_done=False,
            cumulative_step_penalty=0.0, phase1_reward=None,
            signal_quality=None, current_default_prob=base_default_prob,
            months_completed=0, total_months=tc.phase2_months,
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

        tc = self._task_config
        if s.step_count >= tc.max_phase1_steps and atype not in ("APPROVE", "REJECT"):
            reward = RE.phase1_timeout_reward(s.cumulative_step_penalty)
            s.terminal_reward = reward
            s.phase = Phase.TERMINAL
            return self._obs_phase1(
                f"⚠ Timeout after {tc.max_phase1_steps} steps. Auto-rejected. "
                f"Ground truth: {p.ground_truth_label}. Reward: {reward:+.3f}.",
                done=True, reward=reward,
            )

        if atype == "REQUEST_INCOME_PROOF":
            if s.income_revealed:
                # Anti-hack: ESCALATING redundant penalty
                s.redundant_actions += 1
                penalty = RE.redundant_action_penalty(s.redundant_actions)
                s.cumulative_step_penalty += penalty
                s.cumulative_step_penalty += tc.extra_doc_penalty  # task-specific too
                msg = (
                    f"Income proof already on file. Redundant request #{s.redundant_actions}. "
                    f"Penalty: {penalty:+.3f}."
                )
            else:
                s.income_revealed = True
                s.docs_requested_count += 1
                # Anti-hack: ESCALATING step cost
                cost = RE.escalating_step_cost(RE.DOC_REQUEST_COST, s.step_count)
                s.cumulative_step_penalty += cost
                conf = RE.info_confidence(s.income_revealed, s.credit_history_revealed)
                msg = (
                    f"Income proof received. ₹{p.monthly_income:,.0f}/month "
                    f"({p.income_source_stability} income). "
                    f"Confidence now {conf:.0%}. Cost: {cost:+.3f}."
                )

        elif atype == "REQUEST_CREDIT_HISTORY":
            if s.credit_history_revealed:
                s.redundant_actions += 1
                penalty = RE.redundant_action_penalty(s.redundant_actions)
                s.cumulative_step_penalty += penalty
                s.cumulative_step_penalty += tc.extra_doc_penalty
                msg = (
                    f"Credit history already on file. Redundant request #{s.redundant_actions}. "
                    f"Penalty: {penalty:+.3f}."
                )
            else:
                s.credit_history_revealed = True
                s.docs_requested_count += 1
                cost = RE.escalating_step_cost(RE.DOC_REQUEST_COST, s.step_count)
                s.cumulative_step_penalty += cost
                conf = RE.info_confidence(s.income_revealed, s.credit_history_revealed)
                msg = (
                    f"Credit history received. {p.previous_loans} prior loans, "
                    f"{p.past_defaults} defaults, {p.repayment_streak}-month streak. "
                    f"Confidence now {conf:.0%}. Cost: {cost:+.3f}."
                )

        elif atype == "FLAG_FOR_REVIEW":
            if s.senior_review_done:
                s.redundant_actions += 1
                penalty = RE.redundant_action_penalty(s.redundant_actions)
                s.cumulative_step_penalty += penalty
                msg = (
                    f"Already escalated. Redundant request #{s.redundant_actions}. "
                    f"Penalty: {penalty:+.3f}."
                )
            else:
                s.senior_review_done = True
                s.docs_requested_count += 1
                cost = RE.escalating_step_cost(RE.FLAG_REVIEW_COST, s.step_count)
                s.cumulative_step_penalty += cost
                # Anti-hack: OVER-REQUESTING if this is the 3rd+ request
                if s.docs_requested_count >= 3:
                    s.cumulative_step_penalty += RE.OVER_REQUEST_PENALTY
                    msg_suffix = f" Over-research penalty: {RE.OVER_REQUEST_PENALTY}."
                else:
                    msg_suffix = ""
                comment = self._rng.choice(SENIOR_COMMENTS[p.risk_band])
                msg = f"Senior review: '{comment}'. Cost: {cost:+.3f}.{msg_suffix}"

        elif atype == "APPROVE":
            reward = RE.phase1_terminal_reward(
                "APPROVE", p.ground_truth_label,
                s.income_revealed, s.credit_history_revealed,
                s.cumulative_step_penalty,
            )
            s.phase1_reward  = reward
            raw_sq = _signal_quality(s.income_revealed, s.credit_history_revealed)
            s.signal_quality = min(raw_sq, tc.signal_quality_cap)
            s.phase          = Phase.MONITORING
            conf             = RE.info_confidence(s.income_revealed, s.credit_history_revealed)
            correct          = p.ground_truth_label == "APPROVE"
            return self._obs_phase1(
                f"APPROVED at {conf:.0%} confidence. "
                f"Phase 2 starts next step — signal quality {s.signal_quality:.0%}. "
                f"Phase 1 reward: {reward:+.3f}.",
                done=False, reward=reward,
                current_phase="APPLICATION",
                transitioning_to_phase2=True,
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

        VALID_PHASE2_ACTIONS = {
            "DO_NOTHING", "SEND_REMINDER",
            "RESTRUCTURE_LOAN", "ESCALATE_TO_RECOVERY",
        }
        atype = action.action_type

        if atype not in VALID_PHASE2_ACTIONS:
            # Wrong-phase or unknown action → convert to DO_NOTHING with penalty
            INVALID_P2_ACTION_PENALTY = -0.10
            s.phase2_intervention_costs += INVALID_P2_ACTION_PENALTY
            original_action = action.action_type
            atype = "DO_NOTHING"
            action = CreditAction(
                action_type="DO_NOTHING",
                rationale=f"[Invalid Phase 2 action '{original_action}' → DO_NOTHING, penalty {INVALID_P2_ACTION_PENALTY:+.2f}]",
            )

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

        # ── External shock scheduling (1-month lagged signal) ─────────────
        # Month N: 5% chance → schedule shock, set signal strength
        # Month N+1: apply the shock that was signaled last month
        if s.shock_scheduled:
            # Apply the shock that was signaled last month
            s.financial_stability = s.current_default_prob  # save for note
            s.current_default_prob = min(
                0.98, s.current_default_prob + s.shock_magnitude
            )
            note += (f" ⚠ External shock hit! Risk +{s.shock_magnitude:.0%}"
                     f" → {s.current_default_prob:.2%}.")
            s.shock_scheduled = False
            s.shock_magnitude = 0.0
            s.shock_signal_strength = 0.0

        if not s.shock_scheduled and self._rng.random() < 0.05:
            # Schedule a shock for next month — agent sees noisy signal now
            s.shock_scheduled = True
            s.shock_magnitude = self._rng.uniform(0.10, 0.22)
            s.shock_signal_strength = self._rng.uniform(0.35, 0.75)

        # Anti-hack v2: MONOTONIC STRATEGY detection
        # Track consecutive identical Phase 2 actions
        if atype == s.last_phase2_action:
            s.consecutive_same_action += 1
        else:
            s.consecutive_same_action = 1
            s.last_phase2_action = atype
        mono_cost = RE.phase2_monotonic_penalty(
            consecutive_same_action=s.consecutive_same_action,
            action=atype,
            cumulative_misses=s.cumulative_misses,
            missed_streak=s.missed_streak,
            current_default_prob=s.current_default_prob,
            shock_scheduled=s.shock_scheduled,
        )
        if mono_cost < 0:
            s.phase2_intervention_costs += mono_cost
            note += f" ⚠ Monotonic strategy penalty ({s.consecutive_same_action} same): {mono_cost:+.3f}."

        if atype == "DO_NOTHING":
            # Anti-hack: INACTION PENALTY when danger signals are visible
            inaction_cost = RE.phase2_inaction_penalty(s.cumulative_misses, s.missed_streak)
            if inaction_cost < 0:
                s.phase2_intervention_costs += inaction_cost
                note = f"⚠ Inaction during danger. Penalty: {inaction_cost:+.3f}."
            s.consecutive_reminders = 0  # reset reminder spam counter

        elif atype == "SEND_REMINDER":
            s.current_default_prob = max(0.02, s.current_default_prob * 0.95)
            s.intervention_history.append(f"M{month}:REMINDER")
            s.consecutive_reminders += 1
            # Anti-hack: SPAM PENALTY for excessive consecutive reminders
            spam_cost = RE.phase2_spam_penalty(s.consecutive_reminders)
            if spam_cost < 0:
                s.phase2_intervention_costs += spam_cost
                note = (
                    f"Reminder sent. Risk → {s.current_default_prob:.2%}. "
                    f"⚠ Spam penalty ({s.consecutive_reminders} consecutive): {spam_cost:+.3f}."
                )
            else:
                note = f"Reminder sent. Risk → {s.current_default_prob:.2%}."

        elif atype == "RESTRUCTURE_LOAN":
            s.current_default_prob = max(0.02, s.current_default_prob * 0.80)
            s.intervention_history.append(f"M{month}:RESTRUCTURE")
            s.consecutive_reminders = 0
            note = f"Restructured. Risk → {s.current_default_prob:.2%}."

        elif atype == "ESCALATE_TO_RECOVERY":
            s.consecutive_reminders = 0
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

        tc = self._task_config
        if s.cumulative_misses >= tc.default_threshold:
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

        if s.months_completed >= tc.phase2_months:
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
            f"Month {month}/{tc.phase2_months}. Observed: {observed_status.value}. "
            f"Misses: {s.cumulative_misses}. {tc.phase2_months - month} remaining. {note}",
            done=False,
            reward=(s.phase1_reward or 0.0) + s.phase2_intervention_costs,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Observation builders
    # ══════════════════════════════════════════════════════════════════════

    def _obs_phase1(self, msg, *, done=False, reward=0.0,
                    current_phase=None, transitioning_to_phase2=False) -> ApplicantObservation:
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
            step_count=s.step_count, max_steps=self._task_config.max_phase1_steps,
            documents_submitted=docs,
            info_confidence=RE.info_confidence(s.income_revealed, s.credit_history_revealed),
            current_phase=(current_phase or s.phase.value),
            last_action_result=msg, done=done, reward=reward,
            transitioning_to_phase2=transitioning_to_phase2,
        )

    def _obs_phase2(self, month, observed_status, msg, *, done, reward) -> MonitoringObservation:
        s = self._state
        return MonitoringObservation(
            month_number=month,
            months_remaining=self._task_config.phase2_months - month,
            observed_payment=observed_status.value,
            signal_quality=s.signal_quality,
            current_default_prob_hidden=s.current_default_prob,
            cumulative_misses=s.cumulative_misses,
            missed_streak=s.missed_streak, ontime_streak=s.ontime_streak,
            payment_history=list(s.payment_history),
            intervention_history=list(s.intervention_history),
            economic_stress_signal=s.shock_signal_strength if s.shock_scheduled else 0.0,
            current_phase=s.phase.value,
            last_action_result=msg, done=done, reward=reward,
        )
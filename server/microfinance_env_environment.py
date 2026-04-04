"""
microfinance_environment.py — Core RL environment logic.

Episode lifecycle
-----------------
    reset() → initial partial observation (income=None, credit=None)
         ↓
    step(REQUEST_INCOME_PROOF)   → income fields revealed, -0.1 penalty
    step(REQUEST_CREDIT_HISTORY) → credit fields revealed, -0.1 penalty
    step(FLAG_FOR_REVIEW)        → senior comment revealed, -0.15 penalty
         ↓
    step(APPROVE | REJECT)  → terminal step, reward computed, done=True
         OR
    max_steps reached        → forced REJECT, done=True (harsh penalty)

Reward function
---------------
    Correct APPROVE  : +1.0
    Wrong   APPROVE  : -2.0   (false positive — lending to a defaulter)
    Correct REJECT   : +0.6   (not as good as correct approval — inclusion matters)
    Wrong   REJECT   : -1.0   (false negative — denying a creditworthy borrower)
    Each doc request : -0.1   (efficiency cost)
    FLAG_FOR_REVIEW  : -0.15  (heavier — escalation is expensive)
    Max steps hit    : -1.5   (indecisiveness is bad)

The asymmetry in wrong-approval vs wrong-rejection reflects the real-world
stakes: a default harms the institution *and* the borrower's community.
"""

from __future__ import annotations
import random
import uuid
from typing import Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State


# relative imports work when run as part of the package
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import CreditAction, ApplicantObservation, MicrofinanceState
from data_generator import generate_dataset, ApplicantProfile, SENIOR_COMMENTS

# ── Reward constants ───────────────────────────────────────────────────────
R_CORRECT_APPROVE  =  1.0
R_WRONG_APPROVE    = -2.0
R_CORRECT_REJECT   =  0.6
R_WRONG_REJECT     = -1.0
R_DOC_REQUEST      = -0.1
R_FLAG_REVIEW      = -0.15
R_MAX_STEPS        = -1.5

MAX_STEPS_DEFAULT  = 7


class MicrofinanceEnvironment(Environment):
    """
    Microfinance Credit Decision Environment.

    Each episode = one loan application.
    The agent receives partial information and must decide whether to approve
    or reject, optionally requesting documents to reveal hidden fields first.
    """

    def __init__(self, dataset_size: int = 300, seed: int = 42):
        super().__init__()
        self._dataset: list[ApplicantProfile] = generate_dataset(
            n=dataset_size, seed=seed
        )
        self._rng     = random.Random(seed)
        self._ep_state: Optional[MicrofinanceState] = None
        self._profile : Optional[ApplicantProfile]  = None
        self._obs     : Optional[ApplicantObservation] = None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _pick_applicant(self) -> ApplicantProfile:
        return self._rng.choice(self._dataset)

    def _build_initial_obs(self, profile: ApplicantProfile) -> ApplicantObservation:
        """Return partial observation — income and credit history hidden."""
        return ApplicantObservation(
            applicant_id           = profile.applicant_id,
            dependents             = profile.dependents,
            occupation             = profile.occupation,
            loan_amount_requested  = profile.loan_amount_requested,
            region_tier            = profile.region_tier,
            # Hidden until doc requested
            monthly_income         = None,
            income_source_stability= None,
            previous_loans         = None,
            past_defaults          = None,
            repayment_streak       = None,
            senior_review_comment  = None,
            documents_submitted    = [],
            step_count             = 0,
            max_steps              = MAX_STEPS_DEFAULT,
            last_action_result     = (
                "New application received. Income and credit history pending."
            ),
            done   = False,
            reward = 0.0,
        )

    def _senior_comment(self, profile: ApplicantProfile) -> str:
        comments = SENIOR_COMMENTS.get(profile.risk_band, SENIOR_COMMENTS["medium_risk"])
        return self._rng.choice(comments)

    def _compute_terminal_reward(
        self,
        action: str,
        profile: ApplicantProfile,
        step_penalties: float,
    ) -> float:
        if action == "APPROVE":
            base = (
                R_CORRECT_APPROVE
                if profile.ground_truth_label == "APPROVE"
                else R_WRONG_APPROVE
            )
        elif action == "REJECT":
            base = (
                R_CORRECT_REJECT
                if profile.ground_truth_label == "REJECT"
                else R_WRONG_REJECT
            )
        else:
            # max-steps forced rejection
            base = R_MAX_STEPS

        return round(base + step_penalties, 4)

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(self) -> ApplicantObservation:
        profile      = self._pick_applicant()
        self._profile = profile
        obs           = self._build_initial_obs(profile)
        self._obs     = obs

        self._ep_state = MicrofinanceState(
            episode_id             = str(uuid4()),
            step_count             = 0,
            # Hidden ground truth
            ground_truth_label     = profile.ground_truth_label,
            true_default_probability = profile.true_default_probability,
            hidden_monthly_income  = profile.monthly_income,
            hidden_past_defaults   = profile.past_defaults,
            hidden_repayment_streak= profile.repayment_streak,
            hidden_income_stability= profile.income_source_stability,
            income_revealed        = False,
            credit_history_revealed= False,
            senior_review_done     = False,
            cumulative_step_penalty= 0.0,
            terminal_reward        = None,
        )
        return obs

    def step(self, action: CreditAction) -> ApplicantObservation:
        if self._ep_state is None or self._profile is None:
            raise RuntimeError("Call reset() before step().")

        obs     = self._obs
        estate  = self._ep_state
        profile = self._profile
        atype   = action.action_type

        estate.step_count += 1
        obs.step_count     = estate.step_count

        # ── Terminal guard (max steps) ─────────────────────────────────
        if estate.step_count >= MAX_STEPS_DEFAULT and atype not in ("APPROVE", "REJECT"):
            reward = self._compute_terminal_reward(
                "FORCED_REJECT", profile, estate.cumulative_step_penalty
            )
            obs.done   = True
            obs.reward = reward
            obs.last_action_result = (
                f"⚠ Max steps ({MAX_STEPS_DEFAULT}) reached. "
                f"Application auto-rejected. Ground truth: {profile.ground_truth_label}. "
                f"Episode reward: {reward}."
            )
            estate.terminal_reward = reward
            return obs

        # ── Non-terminal actions ──────────────────────────────────────

        if atype == "REQUEST_INCOME_PROOF":
            if estate.income_revealed:
                obs.last_action_result = (
                    "Income proof already submitted. No new information."
                )
                # No extra penalty for redundant request — just a wasted step
            else:
                estate.income_revealed         = True
                estate.cumulative_step_penalty += R_DOC_REQUEST
                obs.monthly_income              = profile.monthly_income
                obs.income_source_stability     = profile.income_source_stability
                obs.documents_submitted.append("income_proof")
                obs.last_action_result = (
                    f"Income proof received. "
                    f"Monthly income: ₹{profile.monthly_income:,.0f}. "
                    f"Stability: {profile.income_source_stability}. "
                    f"Step cost: {R_DOC_REQUEST}."
                )

        elif atype == "REQUEST_CREDIT_HISTORY":
            if estate.credit_history_revealed:
                obs.last_action_result = (
                    "Credit history already on file. No new information."
                )
            else:
                estate.credit_history_revealed  = True
                estate.cumulative_step_penalty += R_DOC_REQUEST
                obs.previous_loans              = profile.previous_loans
                obs.past_defaults               = profile.past_defaults
                obs.repayment_streak            = profile.repayment_streak
                obs.documents_submitted.append("credit_history")
                obs.last_action_result = (
                    f"Credit history received. "
                    f"Previous loans: {profile.previous_loans}, "
                    f"defaults: {profile.past_defaults}, "
                    f"repayment streak: {profile.repayment_streak} months. "
                    f"Step cost: {R_DOC_REQUEST}."
                )

        elif atype == "FLAG_FOR_REVIEW":
            if estate.senior_review_done:
                obs.last_action_result = (
                    "Application already escalated. Senior review complete."
                )
            else:
                estate.senior_review_done       = True
                estate.cumulative_step_penalty += R_FLAG_REVIEW
                obs.senior_review_comment       = self._senior_comment(profile)
                obs.documents_submitted.append("senior_review")
                obs.last_action_result = (
                    f"Senior officer review complete. "
                    f"Comment: '{obs.senior_review_comment}'. "
                    f"Step cost: {R_FLAG_REVIEW}."
                )

        elif atype in ("APPROVE", "REJECT"):
            reward = self._compute_terminal_reward(
                atype, profile, estate.cumulative_step_penalty
            )
            obs.done   = True
            obs.reward = reward
            correct    = (atype == profile.ground_truth_label)
            obs.last_action_result = (
                f"Decision: {atype}. "
                f"Ground truth: {profile.ground_truth_label}. "
                f"{'✓ Correct' if correct else '✗ Incorrect'}. "
                f"Default probability: {profile.true_default_probability:.2f}. "
                f"Episode reward: {reward}."
            )
            estate.terminal_reward = reward

        else:
            obs.last_action_result = f"Unknown action type: {atype}. No state change."

        obs.reward = obs.reward if obs.done else estate.cumulative_step_penalty
        return obs

    @property
    def state(self) -> MicrofinanceState:
        if self._ep_state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._ep_state
"""
client.py — EnvClient for the two-phase Microfinance environment.

The challenge: a single OpenEnv episode can return either ApplicantObservation
(Phase 1) or MonitoringObservation (Phase 2) from each step() call.
We handle this in _parse_result by inspecting the 'current_phase' field
of the payload and routing to the correct dataclass.

Usage (async):
    async with MicrofinanceEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        # Phase 1 — application
        result = await env.step(CreditAction(action_type="REQUEST_INCOME_PROOF"))
        result = await env.step(CreditAction(action_type="REQUEST_CREDIT_HISTORY"))
        result = await env.step(CreditAction(
            action_type="APPROVE",
            rationale="LTI acceptable, no recent defaults"
        ))
        # Phase 2 — monitoring begins
        for month in range(12):
            if result.done:
                break
            result = await env.step(CreditAction(
                action_type="SEND_REMINDER" if result.observation.cumulative_misses > 0
                else "DO_NOTHING"
            ))

Usage (sync):
    with MicrofinanceEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        result = env.step(CreditAction(action_type="REJECT"))
"""

from __future__ import annotations
from typing import Union

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import (
    CreditAction,
    ApplicantObservation,
    MonitoringObservation,
    MicrofinanceState,
    Phase,
)

# Union type for what _parse_result can return
AnyObservation = Union[ApplicantObservation, MonitoringObservation]


class MicrofinanceEnv(EnvClient[CreditAction, AnyObservation, MicrofinanceState]):
    """
    Client for the Microfinance Credit Decision Environment.

    Handles the two-phase observation routing transparently — callers
    can check result.observation.current_phase to know which phase they
    are in and what actions are valid.
    """

    def _step_payload(self, action: CreditAction) -> dict:
        return {
            "action_type": action.action_type,
            "rationale":   action.rationale,
        }

    def _parse_result(self, payload: dict) -> StepResult[AnyObservation]:
        obs_data = payload.get("observation", {})
        phase    = obs_data.get("current_phase", "APPLICATION")

        if phase == "MONITORING":
            obs = MonitoringObservation(
                month_number               = obs_data.get("month_number", 0),
                months_remaining           = obs_data.get("months_remaining", 12),
                observed_payment           = obs_data.get("observed_payment", "ON_TIME"),
                signal_quality             = obs_data.get("signal_quality", 0.9),
                current_default_prob_hidden= obs_data.get("current_default_prob_hidden", 0.0),
                cumulative_misses          = obs_data.get("cumulative_misses", 0),
                missed_streak              = obs_data.get("missed_streak", 0),
                ontime_streak              = obs_data.get("ontime_streak", 0),
                payment_history            = obs_data.get("payment_history", []),
                intervention_history       = obs_data.get("intervention_history", []),
                current_phase              = phase,
                last_action_result         = obs_data.get("last_action_result", ""),
                done                       = payload.get("done", False),
                reward                     = payload.get("reward", 0.0),
            )
        else:
            # APPLICATION or TERMINAL — return ApplicantObservation
            obs = ApplicantObservation(
                applicant_id            = obs_data.get("applicant_id", ""),
                dependents              = obs_data.get("dependents"),
                occupation              = obs_data.get("occupation"),
                loan_amount_requested   = obs_data.get("loan_amount_requested"),
                region_tier             = obs_data.get("region_tier"),
                monthly_income          = obs_data.get("monthly_income"),
                income_source_stability = obs_data.get("income_source_stability"),
                previous_loans          = obs_data.get("previous_loans"),
                past_defaults           = obs_data.get("past_defaults"),
                repayment_streak        = obs_data.get("repayment_streak"),
                senior_review_comment   = obs_data.get("senior_review_comment"),
                step_count              = obs_data.get("step_count", 0),
                max_steps               = obs_data.get("max_steps", 7),
                documents_submitted     = obs_data.get("documents_submitted", []),
                current_phase           = phase,
                last_action_result      = obs_data.get("last_action_result", ""),
                done                    = payload.get("done", False),
                reward                  = payload.get("reward", 0.0),
            )

        return StepResult(
            observation = obs,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> MicrofinanceState:
        return MicrofinanceState(
            episode_id                 = payload.get("episode_id"),
            step_count                 = payload.get("step_count", 0),
            phase                      = Phase(payload.get("phase", "APPLICATION")),
            ground_truth_label         = payload.get("ground_truth_label"),
            true_default_probability   = payload.get("true_default_probability"),
            income_revealed            = payload.get("income_revealed", False),
            credit_history_revealed    = payload.get("credit_history_revealed", False),
            senior_review_done         = payload.get("senior_review_done", False),
            cumulative_step_penalty    = payload.get("cumulative_step_penalty", 0.0),
            phase1_reward              = payload.get("phase1_reward"),
            signal_quality             = payload.get("signal_quality"),
            current_default_prob       = payload.get("current_default_prob", 0.5),
            months_completed           = payload.get("months_completed", 0),
            total_months               = payload.get("total_months", 12),
            payment_history            = payload.get("payment_history", []),
            intervention_history       = payload.get("intervention_history", []),
            missed_streak              = payload.get("missed_streak", 0),
            ontime_streak              = payload.get("ontime_streak", 0),
            cumulative_misses          = payload.get("cumulative_misses", 0),
            terminal_reward            = payload.get("terminal_reward"),
        )
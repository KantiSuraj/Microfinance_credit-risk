"""
client.py — EnvClient wrapper for the Microfinance Credit Decision Environment.

Usage (async):
    async with MicrofinanceEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        result = await env.step(CreditAction(action_type="REQUEST_INCOME_PROOF"))
        result = await env.step(CreditAction(action_type="APPROVE", rationale="..."))

Usage (sync):
    with MicrofinanceEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        result = env.step(CreditAction(action_type="REJECT"))
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import CreditAction, ApplicantObservation, MicrofinanceState


class MicrofinanceEnv(EnvClient[CreditAction, ApplicantObservation, MicrofinanceState]):

    def _step_payload(self, action: CreditAction) -> dict:
        return {
            "action_type": action.action_type,
            "rationale": action.rationale,
        }

    def _parse_result(self, payload: dict) -> StepResult[ApplicantObservation]:
        obs_data = payload.get("observation", {})
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
            last_action_result      = obs_data.get("last_action_result", ""),
            done   = payload.get("done", False),
            reward = payload.get("reward", 0.0),
        )
        return StepResult(
            observation = obs,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> MicrofinanceState:
        return MicrofinanceState(
            episode_id               = payload.get("episode_id"),
            step_count               = payload.get("step_count", 0),
            ground_truth_label       = payload.get("ground_truth_label"),
            true_default_probability = payload.get("true_default_probability"),
            income_revealed          = payload.get("income_revealed", False),
            credit_history_revealed  = payload.get("credit_history_revealed", False),
            senior_review_done       = payload.get("senior_review_done", False),
            cumulative_step_penalty  = payload.get("cumulative_step_penalty", 0.0),
            terminal_reward          = payload.get("terminal_reward"),
        )
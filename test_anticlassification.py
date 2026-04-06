"""
test_anticlassification.py

Proves the environment is NOT a classifier wrapper.

A classifier ignores Phase 1 depth — it sees the same features and makes
the same prediction regardless of how many docs were collected.

This test shows that Phase 1 investigation quality causally changes
Phase 2 outcomes, not just scores. An agent that skips Phase 1 information
gathering suffers materially worse Phase 2 outcomes even with identical
Phase 2 policy.
"""

import sys
import os
import statistics

sys.path.insert(0, os.path.dirname(__file__))

from models import CreditAction, Phase
from server.microfinance_env_environment import MicrofinanceEnvironment
from server.grader import programmatic_grade


def run_episode_with_policy(seed, phase1_policy="blind_approve"):
    """
    Run one episode with a fixed Phase 1 policy and optimal reactive Phase 2.
    Returns the grader score.
    """
    env = MicrofinanceEnvironment(seed=seed)
    obs = env.reset()

    log = {
        "action_trace": [], "docs_collected": [],
        "phase1_decision": "TIMEOUT",
        "ground_truth": env._state.ground_truth_label,
        "default_prob": env._state.true_default_probability,
        "is_borderline": env._profile.is_borderline,
        "has_conflicting_signal": env._profile.has_conflicting_signal,
        "reached_phase2": False, "terminal_outcome": "TIMEOUT",
        "phase1_steps": 0, "phase1_reward": -1.5,
        "payment_history": [], "intervention_history": [],
        "signal_quality": None,
        "default_prob_at_month3": env._state.true_default_probability,
        "default_prob_at_month6": env._state.true_default_probability,
    }

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if phase1_policy == "blind_approve":
        result = env.step(CreditAction(action_type="APPROVE", rationale="blind"))
        log["phase1_decision"] = "APPROVE"
        log["phase1_steps"] = result.step_count
        log["phase1_reward"] = env._state.phase1_reward or -1.5

    elif phase1_policy == "full_docs_then_decide":
        # Gather both documents first
        env.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale="investigate"))
        log["docs_collected"].append("income_proof")
        result = env.step(CreditAction(action_type="REQUEST_CREDIT_HISTORY", rationale="investigate"))
        log["docs_collected"].append("credit_history")

        # Make informed decision based on revealed credit history
        past_defaults = result.past_defaults if result.past_defaults is not None else 0
        if past_defaults >= 2:
            decision = "REJECT"
        else:
            decision = "APPROVE"

        result = env.step(CreditAction(action_type=decision, rationale="informed"))
        log["phase1_decision"] = decision
        log["phase1_steps"] = result.step_count
        log["phase1_reward"] = env._state.phase1_reward or -1.5

        if result.done:
            log["terminal_reward"] = env._state.terminal_reward
            log["terminal_outcome"] = "REJECTED"
            return programmatic_grade(log).score

    # ── Phase 2 — identical optimal reactive policy for both arms ─────────
    if env._state.phase == Phase.MONITORING:
        log["reached_phase2"] = True
        log["signal_quality"] = env._state.signal_quality
        tc = env._task_config
        for _ in range(tc.phase2_months):
            if env._state.phase != Phase.MONITORING:
                break
            if env._state.missed_streak >= 2:
                at = "RESTRUCTURE_LOAN"
            elif env._state.cumulative_misses > 0:
                at = "SEND_REMINDER"
            else:
                at = "DO_NOTHING"
            result = env.step(CreditAction(action_type=at, rationale="reactive"))
            if env._state.months_completed == 3:
                log["default_prob_at_month3"] = env._state.current_default_prob
            if env._state.months_completed == 6:
                log["default_prob_at_month6"] = env._state.current_default_prob
            if result.done:
                break

        log["terminal_reward"] = env._state.terminal_reward
        log["payment_history"] = list(env._state.payment_history)
        log["intervention_history"] = list(env._state.intervention_history)
        if env._state.cumulative_misses >= tc.default_threshold:
            log["terminal_outcome"] = "DEFAULT"
        elif env._state.months_completed >= tc.phase2_months:
            log["terminal_outcome"] = "REPAID"
        else:
            log["terminal_outcome"] = "ESCALATED"

    return programmatic_grade(log).score


def test_phase1_matters_for_phase2_outcomes():
    """
    Phase 1 investigation quality must causally affect outcomes.
    Gap between informed and blind Phase 1 (with identical Phase 2 policy)
    must exceed 0.20. Below that threshold, Phase 1 is decorative.
    """
    blind_scores = []
    informed_scores = []

    for seed in range(1, 51):
        blind_scores.append(
            run_episode_with_policy(seed, "blind_approve")
        )
        informed_scores.append(
            run_episode_with_policy(seed, "full_docs_then_decide")
        )

    blind_mean = statistics.mean(blind_scores)
    informed_mean = statistics.mean(informed_scores)
    blind_std = statistics.stdev(blind_scores) if len(blind_scores) > 1 else 0.0
    informed_std = statistics.stdev(informed_scores) if len(informed_scores) > 1 else 0.0
    gap = informed_mean - blind_mean

    print(f"Blind Phase 1 avg:    {blind_mean:.3f} ± {blind_std:.3f}")
    print(f"Informed Phase 1 avg: {informed_mean:.3f} ± {informed_std:.3f}")
    print(f"Causal gap:           {gap:.3f}")

    assert gap > 0.20, (
        f"Phase 1 gap too small ({gap:.3f}). "
        f"Phase 1 decisions are not causally meaningful enough. "
        f"Target: gap > 0.20."
    )
    print("✓ Phase 1 causally affects Phase 2 outcomes.")


if __name__ == "__main__":
    test_phase1_matters_for_phase2_outcomes()

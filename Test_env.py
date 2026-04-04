"""
test_env.py — Standalone test runner.
Validates environment logic locally WITHOUT needing the OpenEnv HTTP server.
Run: python test_env.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from models import CreditAction
from server.microfinance_env_environment import MicrofinanceEnvironment
from server.grader import programmatic_grade, batch_evaluate

def print_obs(obs, label=""):
    print(f"\n{'─'*60}")
    if label: print(f"  {label}")
    print(f"  Applicant  : {obs.applicant_id}")
    print(f"  Occupation : {obs.occupation} | Region: {obs.region_tier}")
    print(f"  Dependents : {obs.dependents}")
    print(f"  Loan ask   : ₹{obs.loan_amount_requested:,.0f}")
    print(f"  Income     : {'₹{:,.0f}'.format(obs.monthly_income) if obs.monthly_income else 'HIDDEN'}")
    print(f"  Stability  : {obs.income_source_stability or 'HIDDEN'}")
    print(f"  Prev loans : {obs.previous_loans if obs.previous_loans is not None else 'HIDDEN'}")
    print(f"  Defaults   : {obs.past_defaults if obs.past_defaults is not None else 'HIDDEN'}")
    print(f"  Streak     : {obs.repayment_streak if obs.repayment_streak is not None else 'HIDDEN'} months")
    print(f"  Docs done  : {obs.documents_submitted}")
    print(f"  Step       : {obs.step_count}/{obs.max_steps}")
    print(f"  Done       : {obs.done} | Reward: {obs.reward}")
    print(f"  → {obs.last_action_result}")

def run_episode(env, actions, verbose=True):
    obs = env.reset()
    if verbose:
        print_obs(obs, "INITIAL OBSERVATION")
    log = {
        "action_trace"         : [],
        "docs_requested"       : [],
        "decision"             : "TIMEOUT",
        "ground_truth"         : env._ep_state.ground_truth_label,
        "default_prob"         : env._ep_state.true_default_probability,
        "is_borderline"        : env._profile.is_borderline,
        "has_conflicting_signal": env._profile.has_conflicting_signal,
    }
    for action_type, rationale in actions:
        result = env.step(CreditAction(action_type=action_type, rationale=rationale))
        if verbose:
            print_obs(result, f"AFTER: {action_type}")
        log["action_trace"].append({
            "step"      : result.step_count,
            "action"    : action_type,
            "rationale" : rationale,
            "reward_so_far": result.reward,
        })
        if action_type in ("APPROVE", "REJECT"):
            log["decision"] = action_type
        if action_type == "REQUEST_INCOME_PROOF":
            log["docs_requested"].append("income_proof")
        if action_type == "REQUEST_CREDIT_HISTORY":
            log["docs_requested"].append("credit_history")
        if result.done:
            log["steps_taken"]         = result.step_count
            log["cumulative_penalty"]  = env._ep_state.cumulative_step_penalty
            log["terminal_reward"]     = env._ep_state.terminal_reward
            break
    return log


def test_strategic_agent():
    """Agent that requests both docs then decides — should score well."""
    print("\n" + "═"*60)
    print("TEST 1: Strategic agent (requests docs first)")
    print("═"*60)
    env = MicrofinanceEnvironment(seed=7)
    log = run_episode(env, [
        ("REQUEST_INCOME_PROOF",   "Need income to assess LTI ratio"),
        ("REQUEST_CREDIT_HISTORY", "Need defaults history"),
        ("APPROVE",                "LTI within range, no recent defaults"),
    ])
    grade = programmatic_grade(log)
    print(f"\n  Grade: {grade.score:.3f} | Passed: {grade.passed}")
    print(f"  Breakdown: {grade.breakdown}")


def test_impulsive_agent():
    """Agent that approves immediately without requesting any docs."""
    print("\n" + "═"*60)
    print("TEST 2: Impulsive agent (no docs, immediate decision)")
    print("═"*60)
    env = MicrofinanceEnvironment(seed=13)
    log = run_episode(env, [
        ("APPROVE", "Looks fine"),
    ])
    grade = programmatic_grade(log)
    print(f"\n  Grade: {grade.score:.3f} | Passed: {grade.passed}")
    print(f"  Breakdown: {grade.breakdown}")


def test_over_cautious_agent():
    """Agent that requests everything including senior review."""
    print("\n" + "═"*60)
    print("TEST 3: Over-cautious agent (all docs + flag)")
    print("═"*60)
    env = MicrofinanceEnvironment(seed=21)
    log = run_episode(env, [
        ("REQUEST_INCOME_PROOF",   "Need income"),
        ("REQUEST_CREDIT_HISTORY", "Need credit history"),
        ("FLAG_FOR_REVIEW",        "Uncertain, escalating"),
        ("REJECT",                 "Too risky"),
    ])
    grade = programmatic_grade(log)
    print(f"\n  Grade: {grade.score:.3f} | Passed: {grade.passed}")
    print(f"  Breakdown: {grade.breakdown}")


def test_batch_stats():
    """Run 50 random-policy episodes to check reward distribution."""
    import random
    print("\n" + "═"*60)
    print("TEST 4: Random policy — batch statistics")
    print("═"*60)
    env    = MicrofinanceEnvironment(seed=99)
    rng    = random.Random(99)
    logs   = []
    action_pool = [
        "REQUEST_INCOME_PROOF",
        "REQUEST_CREDIT_HISTORY",
        "FLAG_FOR_REVIEW",
        "APPROVE",
        "REJECT",
    ]
    for _ in range(50):
        actions = []
        for _ in range(7):
            a = rng.choice(action_pool)
            actions.append((a, "random"))
            if a in ("APPROVE", "REJECT"):
                break
        log = run_episode(env, actions, verbose=False)
        logs.append(log)

    stats = batch_evaluate(logs)
    print(f"\n  Batch stats over {stats['n_episodes']} episodes:")
    for k, v in stats.items():
        print(f"    {k:30s}: {v}")


if __name__ == "__main__":
    test_strategic_agent()
    test_impulsive_agent()
    test_over_cautious_agent()
    test_batch_stats()
    print("\n✅ All tests complete.")
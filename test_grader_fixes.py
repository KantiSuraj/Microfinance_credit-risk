"""Regression test for grader + environment fixes (Problems 1-5)."""
from server.grader import grade_trajectory
from server.data_generator import generate_applicant, generate_dataset
from server.microfinance_env_environment import MicrofinanceEnvironment
from models import CreditAction
import random

# ══════════════════════════════════════════════════════════════════════
# Problem 1: NoneType guard in grader
# ══════════════════════════════════════════════════════════════════════
log_basic = {
    "phase1_decision": "APPROVE",
    "ground_truth": "APPROVE",
    "default_prob": 0.15,
    "phase1_steps": 3,
    "docs_collected": ["income_proof", "credit_history"],
    "phase1_reward": 0.5,
    "reached_phase2": True,
    "terminal_outcome": "REPAID",
    "terminal_reward": 1.0,
    "payment_history": ["ON_TIME"] * 6,
    "intervention_history": ["M2:REMINDER"],
    "signal_quality": 0.90,
    "is_borderline": False,
    "has_conflicting_signal": False,
    "default_prob_at_month3": 0.12,
    "default_prob_at_month6": None,
}
grade = grade_trajectory(log_basic)
print(f"Problem 1 OK: score={grade.score:.4f} (no NoneType crash)")

# ══════════════════════════════════════════════════════════════════════
# Problem 2: Rejected episode p2 exclusion
# ══════════════════════════════════════════════════════════════════════
log_reject = {
    "phase1_decision": "REJECT",
    "ground_truth": "REJECT",
    "default_prob": 0.75,
    "phase1_steps": 3,
    "docs_collected": ["income_proof", "credit_history"],
    "phase1_reward": 0.8,
    "reached_phase2": False,
    "terminal_outcome": "REJECTED",
    "terminal_reward": 0.8,
    "payment_history": [],
    "intervention_history": [],
    "signal_quality": 0.90,
    "is_borderline": False,
    "has_conflicting_signal": False,
    "default_prob_at_month3": None,
    "default_prob_at_month6": None,
}
grade_r = grade_trajectory(log_reject)
excluded = grade_r.breakdown.get("phase2_excluded", False)
assert excluded, "phase2_excluded should be True when Phase 2 never ran"
assert grade_r.score <= 0.80, f"Reject-only score should be capped at 0.80, got {grade_r.score}"
print(f"Problem 2 OK: score={grade_r.score:.4f} p2_excluded={excluded}")

# ══════════════════════════════════════════════════════════════════════
# Problem 4: APPROVE in Phase 2 should incur penalty, not echo p1 reward
# ══════════════════════════════════════════════════════════════════════
env = MicrofinanceEnvironment(dataset_size=300, seed=42, task_name="basic_lending")
obs = env.reset()

# Phase 1: request income, then approve
env.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale="test"))
obs_approve = env.step(CreditAction(action_type="APPROVE", rationale="test"))
p1_reward = env.state.phase1_reward

# Phase 2: send an APPROVE (invalid)
obs_invalid = env.step(CreditAction(action_type="APPROVE", rationale="invalid repetition"))
p2_cost = env.state.phase2_intervention_costs
running_reward = (p1_reward or 0.0) + p2_cost

print(f"Problem 4: p1_reward={p1_reward:.3f}, p2_intervention_costs={p2_cost:.3f}")
print(f"  Running reward after invalid APPROVE in P2: {running_reward:.3f}")
assert p2_cost < 0, f"Invalid Phase 2 action should incur a penalty, got p2_cost={p2_cost}"
assert running_reward < p1_reward, "Running reward should be less than p1_reward after penalty"
print(f"Problem 4 OK: invalid APPROVE in Phase 2 penalized ({p2_cost:+.3f})")

# ══════════════════════════════════════════════════════════════════════
# Problem 5: Adversarial profiles must always be APPROVE-worthy
# ══════════════════════════════════════════════════════════════════════
# Test 1: generate_applicant with force_adversarial always produces APPROVE
print("\nProblem 5: Checking 200 adversarial profiles...")
n_approve = 0
n_total = 200
for i in range(n_total):
    rng = random.Random(i)
    p = generate_applicant(rng, force_adversarial=True)
    if p.ground_truth_label == "APPROVE":
        n_approve += 1
    else:
        print(f"  FAIL: seed={i} got label={p.ground_truth_label} prob={p.true_default_probability:.3f}")

assert n_approve == n_total, (
    f"All adversarial profiles should be APPROVE, got {n_approve}/{n_total}"
)
print(f"Problem 5a OK: {n_approve}/{n_total} adversarial profiles are APPROVE")

# Test 2: adversarial_portfolio task always selects APPROVE-worthy applicants
env_adv = MicrofinanceEnvironment(dataset_size=300, seed=256, task_name="adversarial_portfolio")
n_approve_env = 0
for _ in range(20):
    env_adv.reset()
    if env_adv.state.ground_truth_label == "APPROVE":
        n_approve_env += 1

assert n_approve_env == 20, (
    f"All adversarial episodes should have ground_truth=APPROVE, got {n_approve_env}/20"
)
print(f"Problem 5b OK: {n_approve_env}/20 adversarial episodes have ground_truth=APPROVE")

print("\n" + "=" * 60)
print("ALL REGRESSION TESTS PASSED")
print("=" * 60)

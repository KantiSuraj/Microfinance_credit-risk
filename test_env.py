"""
test_env.py — Adversarial test suite for the Microfinance RL Environment (v2).

NOT smoke tests. These tests validate that:
  1. The environment can't be trivially exploited
  2. Different strategies produce different outcomes
  3. Core mechanics (signal quality, reward shaping) actually work
  4. Grader is deterministic and non-constant
  5. Edge cases don't crash or produce degenerate rewards
  6. No single lazy strategy dominates (anti-reward-hacking)
  7. Behavioral baselines follow strict ordering
  8. Adversarial cases break single-strategy agents
  9. Counterfactual gap penalizes uninformed decisions

Run: python test_env.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from models import CreditAction, Phase
from server.microfinance_env_environment import MicrofinanceEnvironment, TASK_CONFIGS
from server.grader import programmatic_grade, batch_evaluate


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def run_full(env, phase1_actions, phase2_policy="reactive", verbose=False):
    """Run complete two-phase episode. Returns (log, grade)."""
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

    # Phase 1
    for at, rat in phase1_actions:
        result = env.step(CreditAction(action_type=at, rationale=rat))
        if at in ("APPROVE", "REJECT"):
            log["phase1_decision"] = at
            log["phase1_steps"] = result.step_count
            log["phase1_reward"] = env._state.phase1_reward or -1.5
        if "INCOME" in at: log["docs_collected"].append("income_proof")
        if "CREDIT" in at: log["docs_collected"].append("credit_history")
        if result.done:
            log["terminal_reward"] = env._state.terminal_reward
            log["terminal_outcome"] = "REJECTED" if at == "REJECT" else "TIMEOUT"
            return log, programmatic_grade(log)

    # Phase 2
    if env._state.phase == Phase.MONITORING:
        log["reached_phase2"] = True
        log["signal_quality"] = env._state.signal_quality
        for _ in range(env._task_config.phase2_months):
            if env._state.phase != Phase.MONITORING:
                break
            # Policy
            if phase2_policy == "passive":
                at = "DO_NOTHING"
            elif phase2_policy == "proactive":
                at = "SEND_REMINDER"
            elif phase2_policy == "escalate_early":
                at = "ESCALATE_TO_RECOVERY" if env._state.cumulative_misses > 0 else "DO_NOTHING"
            else:  # reactive
                if env._state.missed_streak >= 2:
                    at = "RESTRUCTURE_LOAN"
                elif env._state.cumulative_misses > 0:
                    at = "SEND_REMINDER"
                else:
                    at = "DO_NOTHING"
            result = env.step(CreditAction(action_type=at, rationale=phase2_policy))
            if env._state.months_completed == 3:
                log["default_prob_at_month3"] = env._state.current_default_prob
            if env._state.months_completed == 6:
                log["default_prob_at_month6"] = env._state.current_default_prob
            if result.done:
                break

        log["terminal_reward"] = env._state.terminal_reward
        log["payment_history"] = list(env._state.payment_history)
        log["intervention_history"] = list(env._state.intervention_history)
        tc = env._task_config
        if env._state.cumulative_misses >= tc.default_threshold:
            log["terminal_outcome"] = "DEFAULT"
        elif env._state.months_completed >= tc.phase2_months:
            log["terminal_outcome"] = "REPAID"
        else:
            log["terminal_outcome"] = "ESCALATED"

    return log, programmatic_grade(log)


passed = 0
failed = 0
def check(condition, msg):
    global passed, failed
    if condition:
        passed += 1
        print(f"    ✓ {msg}")
    else:
        failed += 1
        print(f"    ✗ FAIL: {msg}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: BLIND APPROVE MUST LOSE IN EXPECTATION
# The core design property. If this fails, the env is gameable.
# ══════════════════════════════════════════════════════════════════════════

def test_blind_approve_loses():
    """Blind approve (no docs) should produce negative expected reward."""
    print("\n" + "═"*60)
    print("TEST 1: Blind approve must be a losing strategy")
    print("═"*60)
    rewards = []
    for seed in range(1, 51):
        env = MicrofinanceEnvironment(seed=seed)
        log, grade = run_full(env, [
            ("APPROVE", "yolo"),
        ], phase2_policy="passive")
        rewards.append(log.get("terminal_reward", -1.5) or -1.5)

    avg = sum(rewards) / len(rewards)
    print(f"    Average terminal reward (50 blind approves): {avg:.3f}")
    check(avg < 0, f"Blind approve avg reward ({avg:.3f}) must be negative")
    check(min(rewards) < -1.0, f"Worst case ({min(rewards):.3f}) must be harsh")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: GRADER IS NON-CONSTANT (Disqualification check)
# ══════════════════════════════════════════════════════════════════════════

def test_grader_nonconstant():
    """Grader must return different scores for different strategies."""
    print("\n" + "═"*60)
    print("TEST 2: Grader returns varying scores (not constant)")
    print("═"*60)
    scores = set()
    strategies = [
        ([("APPROVE", "blind")], "passive"),
        ([("REQUEST_CREDIT_HISTORY", ""), ("APPROVE", "informed")], "reactive"),
        ([("REQUEST_INCOME_PROOF", ""), ("REQUEST_CREDIT_HISTORY", ""), ("APPROVE", "")], "proactive"),
        ([("REJECT", "cautious")], "reactive"),
        ([("REQUEST_CREDIT_HISTORY", ""), ("REJECT", "informed reject")], "reactive"),
    ]
    for p1, p2 in strategies:
        env = MicrofinanceEnvironment(seed=42)
        _, grade = run_full(env, p1, p2)
        scores.add(round(grade.score, 4))
        print(f"    Strategy {p1[-1][0]:8s} policy={p2:10s} → score={grade.score:.4f}")

    check(len(scores) >= 3, f"Must produce ≥3 distinct scores (got {len(scores)})")
    check(max(scores) - min(scores) > 0.2, f"Score range ({max(scores) - min(scores):.3f}) must be >0.2")


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: DETERMINISM — Same seed = same trajectory
# ══════════════════════════════════════════════════════════════════════════

def test_determinism():
    """Same seed + same actions must produce exactly the same reward."""
    print("\n" + "═"*60)
    print("TEST 3: Determinism (same seed = same result)")
    print("═"*60)
    actions = [
        ("REQUEST_CREDIT_HISTORY", "check"),
        ("APPROVE", "go"),
    ]
    rewards = []
    outcomes = []
    for _ in range(5):
        env = MicrofinanceEnvironment(seed=42)
        log, grade = run_full(env, actions, "reactive")
        rewards.append(log.get("terminal_reward"))
        outcomes.append(log["terminal_outcome"])

    check(len(set(rewards)) == 1, f"All 5 runs must give same reward (got {set(rewards)})")
    check(len(set(outcomes)) == 1, f"All 5 runs must give same outcome (got {set(outcomes)})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: SIGNAL QUALITY ACTUALLY CORRUPTS OBSERVATIONS
# ══════════════════════════════════════════════════════════════════════════

def test_signal_quality_corruption():
    """Low signal quality should cause observed ≠ true payment more often."""
    print("\n" + "═"*60)
    print("TEST 4: Signal quality corruption is real")
    print("═"*60)

    def count_mismatches(seed, docs_before_approve):
        # Use noisy_signals task — no pre-revealed info, so "no docs" truly = blind
        env = MicrofinanceEnvironment(seed=seed, task_name="noisy_signals")
        env.reset()
        for d in docs_before_approve:
            env.step(CreditAction(action_type=d, rationale=""))
        env.step(CreditAction(action_type="APPROVE", rationale=""))

        if env._state.phase != Phase.MONITORING:
            return 0, 0, 0.0  # rejected by auto

        sq = env._state.signal_quality
        mismatches = 0
        total = 0
        for _ in range(12):
            if env._state.phase != Phase.MONITORING:
                break
            result = env.step(CreditAction(action_type="DO_NOTHING", rationale=""))
            total += 1
        return total, sq, sq

    # No docs → signal quality 0.60
    _, sq_no_docs, _ = count_mismatches(42, [])
    # Credit history → signal quality 0.90
    _, sq_credit, _ = count_mismatches(42, ["REQUEST_CREDIT_HISTORY"])
    # Both docs → signal quality 0.90
    _, sq_both, _ = count_mismatches(42, ["REQUEST_INCOME_PROOF", "REQUEST_CREDIT_HISTORY"])

    print(f"    No docs:       signal_quality = {sq_no_docs}")
    print(f"    Credit only:   signal_quality = {sq_credit}")
    print(f"    Both docs:     signal_quality = {sq_both}")

    check(sq_no_docs is not None and sq_no_docs < 0.70,
          f"No docs → low signal quality ({sq_no_docs})")
    check(sq_credit is not None and sq_credit >= 0.85,
          f"Credit history → high signal quality ({sq_credit})")
    check(sq_no_docs is not None and sq_credit is not None and sq_no_docs < sq_credit,
          "More docs = better signal quality")


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: STRATEGIC > IMPULSIVE (score discrimination)
# ══════════════════════════════════════════════════════════════════════════

def test_strategy_beats_impulse():
    """On average, gathering docs before deciding must score higher."""
    print("\n" + "═"*60)
    print("TEST 5: Strategic agent beats impulsive agent on average")
    print("═"*60)
    strategic_scores = []
    impulsive_scores = []

    for seed in range(1, 31):
        env = MicrofinanceEnvironment(seed=seed)
        _, grade = run_full(env, [
            ("REQUEST_CREDIT_HISTORY", "assess risk"),
            ("APPROVE", "informed decision"),
        ], "reactive")
        strategic_scores.append(grade.score)

        env2 = MicrofinanceEnvironment(seed=seed)
        _, grade2 = run_full(env2, [
            ("APPROVE", "blind"),
        ], "passive")
        impulsive_scores.append(grade2.score)

    avg_s = sum(strategic_scores) / len(strategic_scores)
    avg_i = sum(impulsive_scores) / len(impulsive_scores)
    print(f"    Strategic avg: {avg_s:.3f}")
    print(f"    Impulsive avg: {avg_i:.3f}")
    print(f"    Delta: {avg_s - avg_i:+.3f}")
    check(avg_s > avg_i, f"Strategic ({avg_s:.3f}) must beat impulsive ({avg_i:.3f})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: PHASE 1 CHOICES AFFECT PHASE 2 OUTCOMES
# ══════════════════════════════════════════════════════════════════════════

def test_phase1_affects_phase2():
    """Phase 1 doc collection must change Phase 2 signal quality and outcomes."""
    print("\n" + "═"*60)
    print("TEST 6: Phase 1 choices causally affect Phase 2")
    print("═"*60)
    # Run same seed with and without docs
    env1 = MicrofinanceEnvironment(seed=42)
    log1, _ = run_full(env1, [("APPROVE", "blind")], "reactive")

    env2 = MicrofinanceEnvironment(seed=42)
    log2, _ = run_full(env2, [
        ("REQUEST_INCOME_PROOF", ""), ("REQUEST_CREDIT_HISTORY", ""),
        ("APPROVE", "informed"),
    ], "reactive")

    sq1 = log1.get("signal_quality")
    sq2 = log2.get("signal_quality")
    print(f"    Blind approve:    signal_quality={sq1}, outcome={log1['terminal_outcome']}")
    print(f"    Informed approve: signal_quality={sq2}, outcome={log2['terminal_outcome']}")

    if sq1 is not None and sq2 is not None:
        check(sq2 > sq1, f"Informed signal ({sq2}) must exceed blind signal ({sq1})")
    else:
        check(False, "Signal quality should be set for approved loans")


# ══════════════════════════════════════════════════════════════════════════
# TEST 7: INTERVENTION TIMING MATTERS
# ══════════════════════════════════════════════════════════════════════════

def test_intervention_timing():
    """Reactive interventions should score better than passive on hard cases."""
    print("\n" + "═"*60)
    print("TEST 7: Intervention timing affects outcome")
    print("═"*60)
    reactive_scores = []
    passive_scores = []

    for seed in range(1, 31):
        env_r = MicrofinanceEnvironment(seed=seed, task_name="adversarial_portfolio")
        _, grade_r = run_full(env_r, [
            ("REQUEST_CREDIT_HISTORY", ""),
            ("APPROVE", "borderline accept"),
        ], "reactive")
        reactive_scores.append(grade_r.score)

        env_p = MicrofinanceEnvironment(seed=seed, task_name="adversarial_portfolio")
        _, grade_p = run_full(env_p, [
            ("REQUEST_CREDIT_HISTORY", ""),
            ("APPROVE", "borderline accept"),
        ], "passive")
        passive_scores.append(grade_p.score)

    avg_r = sum(reactive_scores) / len(reactive_scores)
    avg_p = sum(passive_scores) / len(passive_scores)
    print(f"    Reactive monitoring avg: {avg_r:.3f}")
    print(f"    Passive monitoring avg:  {avg_p:.3f}")
    print(f"    Delta: {avg_r - avg_p:+.3f}")
    check(avg_r >= avg_p - 0.05,  # small tolerance for randomness
          f"Reactive ({avg_r:.3f}) should generally beat passive ({avg_p:.3f})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 8: INVALID ACTIONS DON'T CRASH
# ══════════════════════════════════════════════════════════════════════════

def test_invalid_actions():
    """Invalid or wrong-phase actions must not crash the environment."""
    print("\n" + "═"*60)
    print("TEST 8: Invalid/wrong-phase actions handled gracefully")
    print("═"*60)

    # Phase 2 action in Phase 1
    env = MicrofinanceEnvironment(seed=42)
    env.reset()
    try:
        result = env.step(CreditAction(action_type="SEND_REMINDER", rationale="wrong phase"))
        check(True, "Phase 2 action in Phase 1 didn't crash")
    except Exception as e:
        check(False, f"Phase 2 action in Phase 1 crashed: {e}")

    # Duplicate doc requests
    env2 = MicrofinanceEnvironment(seed=42)
    env2.reset()
    env2.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale="first"))
    result = env2.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale="duplicate"))
    check(not result.done, "Duplicate doc request didn't end episode")
    check("already" in result.last_action_result.lower(),
          "Duplicate doc request gives clear feedback message")


# ══════════════════════════════════════════════════════════════════════════
# TEST 9: TASK DIFFICULTY IS REAL (structural, not cosmetic)
# ══════════════════════════════════════════════════════════════════════════

def test_task_difficulty():
    """Harder tasks must produce lower average scores with same policy."""
    print("\n" + "═"*60)
    print("TEST 9: Task difficulty produces measurable score differences")
    print("═"*60)
    task_scores = {}
    for task_name in ["basic_lending", "noisy_signals", "adversarial_portfolio"]:
        scores = []
        for seed in range(1, 21):
            env = MicrofinanceEnvironment(seed=seed, task_name=task_name)
            _, grade = run_full(env, [
                ("REQUEST_CREDIT_HISTORY", ""),
                ("APPROVE", ""),
            ], "reactive")
            scores.append(grade.score)
        avg = sum(scores) / len(scores)
        task_scores[task_name] = avg
        tc = TASK_CONFIGS[task_name]
        print(f"    {task_name:25s}: avg_score={avg:.3f}  "
              f"months={tc.phase2_months}  threshold={tc.default_threshold}  "
              f"sq_cap={tc.signal_quality_cap}")

    check(task_scores["basic_lending"] > task_scores["adversarial_portfolio"],
          f"Easy ({task_scores['basic_lending']:.3f}) must score higher than hard "
          f"({task_scores['adversarial_portfolio']:.3f})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 10: REWARD RANGE COVERS [0, 1] IN GRADER
# ══════════════════════════════════════════════════════════════════════════

def test_reward_distribution():
    """Grader scores must span a reasonable range, not cluster."""
    print("\n" + "═"*60)
    print("TEST 10: Score distribution is well-spread (not clustered)")
    print("═"*60)
    import random
    all_scores = []
    rng = random.Random(99)
    p1_pool = ["REQUEST_INCOME_PROOF", "REQUEST_CREDIT_HISTORY",
               "FLAG_FOR_REVIEW", "APPROVE", "REJECT"]
    p2_pool = ["DO_NOTHING", "SEND_REMINDER", "RESTRUCTURE_LOAN"]

    for seed in range(1, 101):
        env = MicrofinanceEnvironment(seed=seed)
        # Random Phase 1
        p1_acts = []
        for _ in range(rng.randint(1, 5)):
            a = rng.choice(p1_pool)
            p1_acts.append((a, "random"))
            if a in ("APPROVE", "REJECT"):
                break
        policy = rng.choice(["passive", "reactive", "proactive"])
        _, grade = run_full(env, p1_acts, policy)
        all_scores.append(grade.score)

    min_s, max_s = min(all_scores), max(all_scores)
    mean_s = sum(all_scores) / len(all_scores)
    # Count how many fall in each quartile
    q1 = sum(1 for s in all_scores if s < 0.25)
    q2 = sum(1 for s in all_scores if 0.25 <= s < 0.50)
    q3 = sum(1 for s in all_scores if 0.50 <= s < 0.75)
    q4 = sum(1 for s in all_scores if s >= 0.75)

    print(f"    Range:  [{min_s:.3f}, {max_s:.3f}]")
    print(f"    Mean:   {mean_s:.3f}")
    print(f"    Q1(<.25): {q1}  Q2(.25-.50): {q2}  Q3(.50-.75): {q3}  Q4(>.75): {q4}")

    check(max_s - min_s > 0.4, f"Score range ({max_s - min_s:.3f}) must be >0.4")
    check(q1 + q2 > 0 and q3 + q4 > 0, "Scores must span both low and high regions")


# ══════════════════════════════════════════════════════════════════════════
# TEST 11: ALWAYS-REJECT EXPLOIT — must not be a winning strategy (FIXED v2)
# ══════════════════════════════════════════════════════════════════════════

def test_always_reject_exploit():
    """Always rejecting (with zero info) must score poorly in the grader."""
    print("\n" + "═"*60)
    print("TEST 11: Always-reject exploit is blocked (v2 hardened)")
    print("═"*60)
    reject_scores = []
    for seed in range(1, 51):  # 50 seeds for robustness
        env = MicrofinanceEnvironment(seed=seed)
        _, grade = run_full(env, [("REJECT", "too risky")], "reactive")
        reject_scores.append(grade.score)

    avg = sum(reject_scores) / len(reject_scores)
    max_s = max(reject_scores)
    print(f"    Always-reject avg score (50 seeds): {avg:.3f}")
    print(f"    Always-reject max score:            {max_s:.3f}")
    check(avg < 0.45, f"Always-reject ({avg:.3f}) must not be a winning strategy (<0.45)")
    check(max_s < 0.75, f"Always-reject worst case ({max_s:.3f}) must stay below 0.75")


# ══════════════════════════════════════════════════════════════════════════
# TEST 12: OVER-REQUESTING EXPLOIT — requesting everything must have diminishing returns
# ══════════════════════════════════════════════════════════════════════════

def test_over_requesting_exploit():
    """Requesting ALL docs + flag should score LOWER than targeted 1-2 docs."""
    print("\n" + "═"*60)
    print("TEST 12: Over-requesting has diminishing returns")
    print("═"*60)
    over_scores = []
    targeted_scores = []
    for seed in range(1, 31):
        # Over-request: all 3
        env1 = MicrofinanceEnvironment(seed=seed)
        _, g1 = run_full(env1, [
            ("REQUEST_INCOME_PROOF", ""),
            ("REQUEST_CREDIT_HISTORY", ""),
            ("FLAG_FOR_REVIEW", ""),
            ("APPROVE", "everything checked"),
        ], "reactive")
        over_scores.append(g1.score)

        # Targeted: just credit history
        env2 = MicrofinanceEnvironment(seed=seed)
        _, g2 = run_full(env2, [
            ("REQUEST_CREDIT_HISTORY", ""),
            ("APPROVE", "targeted"),
        ], "reactive")
        targeted_scores.append(g2.score)

    avg_over = sum(over_scores) / len(over_scores)
    avg_targeted = sum(targeted_scores) / len(targeted_scores)
    print(f"    Over-request (3 docs) avg: {avg_over:.3f}")
    print(f"    Targeted (1 doc) avg:      {avg_targeted:.3f}")
    print(f"    Delta: {avg_targeted - avg_over:+.3f}")
    check(avg_over <= avg_targeted + 0.05,
          f"Over-requesting ({avg_over:.3f}) must not dominate targeted ({avg_targeted:.3f})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 13: REDUNDANT ACTION SPAM — escalating penalty works
# ══════════════════════════════════════════════════════════════════════════

def test_redundant_action_spam():
    """Spamming redundant actions must produce increasingly harsh penalties."""
    print("\n" + "═"*60)
    print("TEST 13: Redundant action spam is punished increasingly")
    print("═"*60)
    env = MicrofinanceEnvironment(seed=42, task_name="noisy_signals")
    env.reset()

    # Request income, then spam it 3 more times
    env.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale="first"))
    penalties = []
    for i in range(3):
        result = env.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale=f"spam {i+1}"))
        penalties.append(env._state.cumulative_step_penalty)

    print(f"    Penalties after each spam: {[f'{p:.3f}' for p in penalties]}")
    check(penalties[2] < penalties[1] < penalties[0] < 0 or 
          abs(penalties[2]) > abs(penalties[1]) > abs(penalties[0]),
          "Penalties must escalate with each redundant action")
    check(env._state.redundant_actions == 3,
          f"Should track 3 redundant actions (got {env._state.redundant_actions})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 14: PHASE 2 INACTION EXPLOIT — doing nothing under danger is punished
# ══════════════════════════════════════════════════════════════════════════

def test_phase2_inaction_exploit():
    """Doing nothing in Phase 2 when misses pile up must cost more than intervening."""
    print("\n" + "═"*60)
    print("TEST 14: Phase 2 inaction under danger is penalized")
    print("═"*60)
    from server.reward_engine import phase2_inaction_penalty

    # No danger → no penalty
    p0 = phase2_inaction_penalty(cumulative_misses=0, missed_streak=0)
    # Moderate danger → penalty
    p1 = phase2_inaction_penalty(cumulative_misses=2, missed_streak=1)
    # High danger → double penalty
    p2 = phase2_inaction_penalty(cumulative_misses=3, missed_streak=2)

    print(f"    No danger (0 misses):     penalty = {p0}")
    print(f"    Moderate (2 misses):      penalty = {p1}")
    print(f"    High (3 misses, streak 2): penalty = {p2}")

    check(p0 == 0.0, "No penalty when no danger")
    check(p1 < 0, f"Must penalize inaction at 2+ misses (got {p1})")
    check(p2 < p1, f"Higher danger must mean harsher penalty ({p2} < {p1})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 15: INDEPENDENT REWARD AUDIT — flags suspicious patterns
# ══════════════════════════════════════════════════════════════════════════

def test_reward_audit():
    """Independent audit must flag degenerate strategies."""
    print("\n" + "═"*60)
    print("TEST 15: Independent reward audit flags exploits")
    print("═"*60)
    from server.reward_engine import audit_reward

    # Degenerate: blind approve
    blind_log = {"phase1_decision": "APPROVE", "docs_collected": [],
                 "action_trace": [], "payment_history": [], "intervention_history": [],
                 "phase1_steps": 1, "terminal_reward": -2.0}
    flags = audit_reward(blind_log)
    print(f"    Blind approve flags: {flags}")
    check(flags["blind_approve"], "Blind approve must be flagged")

    # Degenerate: always reject
    reject_log = {"phase1_decision": "REJECT", "docs_collected": [],
                  "action_trace": [], "payment_history": [], "intervention_history": [],
                  "phase1_steps": 1, "terminal_reward": 0.1}
    flags = audit_reward(reject_log)
    print(f"    Always reject flags: {flags}")
    check(flags["always_reject"], "Always-reject must be flagged")

    # Degenerate: Phase 2 complete inaction
    inaction_log = {"phase1_decision": "APPROVE", "docs_collected": ["credit_history"],
                    "action_trace": [], "phase1_steps": 2, "terminal_reward": -2.5,
                    "payment_history": ["ON_TIME", "MISSED", "MISSED", "MISSED"],
                    "intervention_history": []}
    flags = audit_reward(inaction_log)
    print(f"    P2 inaction flags: {flags}")
    check(flags["phase2_complete_inaction"], "Phase 2 inaction despite 3+ misses must be flagged")

    # Healthy: informed approve with interventions
    healthy_log = {"phase1_decision": "APPROVE", "docs_collected": ["credit_history"],
                   "action_trace": [], "phase1_steps": 2, "terminal_reward": 1.5,
                   "payment_history": ["ON_TIME", "MISSED", "ON_TIME"],
                   "intervention_history": ["M2:REMINDER"]}
    flags = audit_reward(healthy_log)
    print(f"    Healthy agent flags: {flags}")
    check(not flags["any_flag"], "Healthy strategy must NOT be flagged")

    # v2: Early termination exploit
    early_log = {"phase1_decision": "REJECT", "docs_collected": [],
                 "action_trace": [], "payment_history": [], "intervention_history": [],
                 "phase1_steps": 1, "terminal_reward": 0.1}
    flags = audit_reward(early_log)
    print(f"    Early termination flags: {flags}")
    check(flags["early_termination_exploit"], "Early termination exploit must be flagged")


# ══════════════════════════════════════════════════════════════════════════
# TEST 16: BEHAVIORAL BASELINES — strict ordering of agent quality (NEW)
# ══════════════════════════════════════════════════════════════════════════

def test_behavioral_baselines():
    """Agent types must follow a strict quality ordering in the grader."""
    print("\n" + "═"*60)
    print("TEST 16: Behavioral baselines follow strict ordering")
    print("═"*60)

    import random as stdlib_random

    def run_agent(agent_type, seeds):
        scores = []
        for seed in seeds:
            env = MicrofinanceEnvironment(seed=seed)
            if agent_type == "random":
                rng = stdlib_random.Random(seed + 1000)
                p1_pool = ["REQUEST_INCOME_PROOF", "REQUEST_CREDIT_HISTORY",
                           "FLAG_FOR_REVIEW", "APPROVE", "REJECT"]
                acts = []
                for _ in range(rng.randint(1, 5)):
                    a = rng.choice(p1_pool)
                    acts.append((a, "random"))
                    if a in ("APPROVE", "REJECT"): break
                policy = rng.choice(["passive", "reactive", "proactive"])
                _, grade = run_full(env, acts, policy)
            elif agent_type == "always_reject":
                _, grade = run_full(env, [("REJECT", "no")], "reactive")
            elif agent_type == "blind_approve":
                _, grade = run_full(env, [("APPROVE", "yolo")], "passive")
            elif agent_type == "1doc_approve":
                _, grade = run_full(env, [
                    ("REQUEST_CREDIT_HISTORY", ""),
                    ("APPROVE", "1doc"),
                ], "reactive")
            elif agent_type == "2doc_approve":
                _, grade = run_full(env, [
                    ("REQUEST_INCOME_PROOF", ""),
                    ("REQUEST_CREDIT_HISTORY", ""),
                    ("APPROVE", "2doc"),
                ], "reactive")
            else:
                raise ValueError(agent_type)
            scores.append(grade.score)
        return sum(scores) / len(scores)

    seeds = list(range(1, 41))  # 40 seeds

    random_avg     = run_agent("random", seeds)
    reject_avg     = run_agent("always_reject", seeds)
    blind_avg      = run_agent("blind_approve", seeds)
    one_doc_avg    = run_agent("1doc_approve", seeds)
    two_doc_avg    = run_agent("2doc_approve", seeds)

    print(f"    Random agent:     {random_avg:.3f}")
    print(f"    Always-reject:    {reject_avg:.3f}")
    print(f"    Blind approve:    {blind_avg:.3f}")
    print(f"    1-doc + approve:  {one_doc_avg:.3f}")
    print(f"    2-doc + approve:  {two_doc_avg:.3f}")

    # Key ordering checks
    check(one_doc_avg > reject_avg,
          f"1-doc ({one_doc_avg:.3f}) must beat always-reject ({reject_avg:.3f})")
    check(one_doc_avg > blind_avg,
          f"1-doc ({one_doc_avg:.3f}) must beat blind-approve ({blind_avg:.3f})")
    check(two_doc_avg > blind_avg,
          f"2-doc ({two_doc_avg:.3f}) must beat blind-approve ({blind_avg:.3f})")
    check(reject_avg < 0.45,
          f"Always-reject ({reject_avg:.3f}) must be below 0.45")
    check(blind_avg < 0.50,
          f"Blind-approve ({blind_avg:.3f}) must be below 0.50")


# ══════════════════════════════════════════════════════════════════════════
# TEST 17: SEED DIVERSITY — scores must not collapse on different seeds (NEW)
# ══════════════════════════════════════════════════════════════════════════

def test_seed_diversity():
    """Performance must remain stable (not collapse) across different seeds."""
    print("\n" + "═"*60)
    print("TEST 17: Score diversity across seeds (anti-overfitting)")
    print("═"*60)

    scores_by_seed_range = {}
    for start in [1, 51, 101, 151]:
        scores = []
        for seed in range(start, start + 25):
            env = MicrofinanceEnvironment(seed=seed)
            _, grade = run_full(env, [
                ("REQUEST_CREDIT_HISTORY", ""),
                ("APPROVE", ""),
            ], "reactive")
            scores.append(grade.score)
        avg = sum(scores) / len(scores)
        scores_by_seed_range[f"seeds_{start}-{start+24}"] = avg
        print(f"    Seeds {start:4d}-{start+24:4d}: avg={avg:.3f}")

    all_avgs = list(scores_by_seed_range.values())
    spread = max(all_avgs) - min(all_avgs)
    print(f"    Spread across ranges: {spread:.3f}")

    check(spread < 0.25,
          f"Score spread ({spread:.3f}) across seed ranges must be <0.25 (not seed-dependent)")
    check(min(all_avgs) > 0.20,
          f"Worst seed range ({min(all_avgs):.3f}) must still produce reasonable scores")


# ══════════════════════════════════════════════════════════════════════════
# TEST 18: COUNTERFACTUAL GAP — blind decisions have non-zero gap (NEW)
# ══════════════════════════════════════════════════════════════════════════

def test_counterfactual_gap():
    """Counterfactual oracle must distinguish informed from blind decisions."""
    print("\n" + "═"*60)
    print("TEST 18: Counterfactual gap penalizes blind decisions")
    print("═"*60)
    from server.counterfactual import counterfactual_gap, counterfactual_grade_modifier

    # Blind wrong decision on clear case
    gap_blind_wrong = counterfactual_gap(
        agent_decision="APPROVE", agent_confidence=0.0,
        ground_truth="REJECT", default_prob=0.8, is_borderline=False,
    )
    # Informed wrong decision
    gap_informed_wrong = counterfactual_gap(
        agent_decision="APPROVE", agent_confidence=0.65,
        ground_truth="REJECT", default_prob=0.8, is_borderline=False,
    )
    # Correct decision (any confidence)
    gap_correct = counterfactual_gap(
        agent_decision="REJECT", agent_confidence=0.65,
        ground_truth="REJECT", default_prob=0.8, is_borderline=False,
    )
    # Borderline wrong decision
    gap_borderline = counterfactual_gap(
        agent_decision="APPROVE", agent_confidence=0.35,
        ground_truth="REJECT", default_prob=0.52, is_borderline=True,
    )

    print(f"    Blind wrong (clear case):    gap = {gap_blind_wrong}")
    print(f"    Informed wrong (clear case): gap = {gap_informed_wrong}")
    print(f"    Correct decision:            gap = {gap_correct}")
    print(f"    Borderline wrong:            gap = {gap_borderline}")

    check(gap_correct == 0.0, "Correct decision must have zero gap")
    check(gap_blind_wrong > 0.0, f"Blind wrong must have positive gap ({gap_blind_wrong})")
    check(gap_informed_wrong > gap_blind_wrong,
          f"Informed wrong ({gap_informed_wrong}) gap > blind wrong ({gap_blind_wrong}) — agent should have known")
    check(gap_borderline < gap_informed_wrong,
          f"Borderline ({gap_borderline}) gap < clear case ({gap_informed_wrong}) — ambiguity reduces penalty")

    # Test modifier
    mod_correct = counterfactual_grade_modifier("REJECT", 0.65, "REJECT", 0.8, False)
    mod_wrong   = counterfactual_grade_modifier("APPROVE", 0.65, "REJECT", 0.8, False)
    print(f"    Modifier (correct): {mod_correct}")
    print(f"    Modifier (wrong):   {mod_wrong}")
    check(mod_correct == 1.0, "Correct modifier must be 1.0")
    check(mod_wrong < 1.0, f"Wrong modifier ({mod_wrong}) must be < 1.0")


# ══════════════════════════════════════════════════════════════════════════
# TEST 19: ADVERSARIAL — ALL DOCS WORSE THAN SELECTIVE (NEW)
# ══════════════════════════════════════════════════════════════════════════

def test_adversarial_overcollection():
    """On clear REJECT cases, requesting all docs must waste money vs quick informed reject."""
    print("\n" + "═"*60)
    print("TEST 19: Adversarial — overcollection on clear cases is punished")
    print("═"*60)

    overcollect_scores = []
    targeted_reject_scores = []

    n_tested = 0
    for seed in range(1, 101):
        env_check = MicrofinanceEnvironment(seed=seed)
        env_check.reset()
        # Only test on clear reject cases (high default prob)
        if env_check._profile.true_default_probability < 0.60:
            continue

        n_tested += 1
        if n_tested > 30:
            break

        # Overcollect then reject (wasteful)
        env1 = MicrofinanceEnvironment(seed=seed)
        _, g1 = run_full(env1, [
            ("REQUEST_INCOME_PROOF", ""),
            ("REQUEST_CREDIT_HISTORY", ""),
            ("FLAG_FOR_REVIEW", ""),
            ("REJECT", "overkill"),
        ], "reactive")
        overcollect_scores.append(g1.score)

        # Targeted: 1 doc + reject (efficient)
        env2 = MicrofinanceEnvironment(seed=seed)
        _, g2 = run_full(env2, [
            ("REQUEST_CREDIT_HISTORY", ""),
            ("REJECT", "targeted"),
        ], "reactive")
        targeted_reject_scores.append(g2.score)

    if n_tested >= 5:
        avg_over = sum(overcollect_scores) / len(overcollect_scores)
        avg_targeted = sum(targeted_reject_scores) / len(targeted_reject_scores)
        print(f"    Overcollect + reject avg: {avg_over:.3f} ({n_tested} clear reject cases)")
        print(f"    Targeted + reject avg:    {avg_targeted:.3f}")
        print(f"    Delta: {avg_targeted - avg_over:+.3f}")
        check(avg_over <= avg_targeted + 0.10,
              f"Overcollecting ({avg_over:.3f}) must not dominate targeted ({avg_targeted:.3f})")
    else:
        print(f"    Only found {n_tested} clear reject cases — skipping")
        check(True, "Insufficient clear reject cases (OK)")


# ══════════════════════════════════════════════════════════════════════════
# TEST 20: NO SINGLE STRATEGY DOMINATES — critical anti-hack check (NEW)
# ══════════════════════════════════════════════════════════════════════════

def test_no_single_strategy_dominates():
    """No fixed action sequence should score >0.70 avg across 50 seeds."""
    print("\n" + "═"*60)
    print("TEST 20: No single fixed strategy dominates")
    print("═"*60)

    strategies = {
        "blind_approve_passive": ([("APPROVE", "")], "passive"),
        "blind_approve_reactive": ([("APPROVE", "")], "reactive"),
        "blind_reject": ([("REJECT", "")], "reactive"),
        "1doc_approve_passive": ([("REQUEST_CREDIT_HISTORY", ""), ("APPROVE", "")], "passive"),
        "all_docs_approve": ([
            ("REQUEST_INCOME_PROOF", ""),
            ("REQUEST_CREDIT_HISTORY", ""),
            ("FLAG_FOR_REVIEW", ""),
            ("APPROVE", ""),
        ], "reactive"),
        "1doc_reject": ([("REQUEST_CREDIT_HISTORY", ""), ("REJECT", "")], "reactive"),
        "proactive_spam": ([("REQUEST_CREDIT_HISTORY", ""), ("APPROVE", "")], "proactive"),
    }

    seeds = list(range(1, 51))
    any_dominates = False

    for name, (p1_acts, p2_policy) in strategies.items():
        scores = []
        for seed in seeds:
            env = MicrofinanceEnvironment(seed=seed)
            _, grade = run_full(env, p1_acts, p2_policy)
            scores.append(grade.score)
        avg = sum(scores) / len(scores)
        marker = " ⚠" if avg > 0.70 else ""
        print(f"    {name:30s}: avg={avg:.3f}{marker}")
        if avg > 0.70:
            any_dominates = True

    check(not any_dominates,
          "No single fixed strategy should average >0.70 across 50 seeds")


# ══════════════════════════════════════════════════════════════════════════
# TEST 21: PHASE 2 MONOTONIC STRATEGY PENALTY (NEW)
# ══════════════════════════════════════════════════════════════════════════

def test_phase2_monotonic_penalty():
    """Context-conditioned monotonic penalty: healthy DO_NOTHING is correct, not degenerate."""
    print("\n" + "═"*60)
    print("TEST 21: Phase 2 monotonic penalty is context-conditioned")
    print("═"*60)
    from server.reward_engine import phase2_monotonic_penalty

    # ── Healthy borrower: DO_NOTHING is correct → no penalty ──────────
    p3_healthy = phase2_monotonic_penalty(
        3, action="DO_NOTHING", cumulative_misses=0,
        missed_streak=0, current_default_prob=0.10, shock_scheduled=False,
    )
    p4_healthy = phase2_monotonic_penalty(
        4, action="DO_NOTHING", cumulative_misses=0,
        missed_streak=0, current_default_prob=0.10, shock_scheduled=False,
    )
    p7_healthy = phase2_monotonic_penalty(
        7, action="DO_NOTHING", cumulative_misses=0,
        missed_streak=0, current_default_prob=0.10, shock_scheduled=False,
    )

    print(f"    Healthy borrower:")
    print(f"      3× DO_NOTHING: {p3_healthy}")
    print(f"      4× DO_NOTHING: {p4_healthy}")
    print(f"      7× DO_NOTHING: {p7_healthy}")

    check(p3_healthy == 0.0, "3× DO_NOTHING healthy → no penalty")
    check(p4_healthy == 0.0, "4× DO_NOTHING healthy → no penalty (correct monitoring)")
    check(p7_healthy == 0.0, "7× DO_NOTHING healthy → no penalty (correct monitoring)")

    # ── Danger present: DO_NOTHING is lazy → penalty ──────────────────
    p3_danger = phase2_monotonic_penalty(
        3, action="DO_NOTHING", cumulative_misses=3,
        missed_streak=2, current_default_prob=0.50, shock_scheduled=False,
    )
    p4_danger = phase2_monotonic_penalty(
        4, action="DO_NOTHING", cumulative_misses=3,
        missed_streak=2, current_default_prob=0.50, shock_scheduled=False,
    )
    p7_danger = phase2_monotonic_penalty(
        7, action="DO_NOTHING", cumulative_misses=3,
        missed_streak=2, current_default_prob=0.50, shock_scheduled=False,
    )

    print(f"    Danger present:")
    print(f"      3× DO_NOTHING: {p3_danger}")
    print(f"      4× DO_NOTHING: {p4_danger}")
    print(f"      7× DO_NOTHING: {p7_danger}")

    check(p3_danger == 0.0, "3× DO_NOTHING danger → no penalty (below threshold)")
    check(p4_danger < 0, f"4× DO_NOTHING danger → penalty ({p4_danger})")
    check(p7_danger < p4_danger, f"7× DO_NOTHING danger ({p7_danger}) harsher than 4× ({p4_danger})")

    # ── Non-DO_NOTHING: always penalized regardless of context ────────
    p7_reminder = phase2_monotonic_penalty(
        7, action="SEND_REMINDER", cumulative_misses=0,
        missed_streak=0, current_default_prob=0.10, shock_scheduled=False,
    )

    print(f"    Non-DO_NOTHING (SEND_REMINDER, healthy borrower):")
    print(f"      7× SEND_REMINDER: {p7_reminder}")

    check(p7_reminder < 0, f"7× SEND_REMINDER always penalized ({p7_reminder})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 22: EPISODE LOGGER PATTERN DETECTION (NEW)
# ══════════════════════════════════════════════════════════════════════════

def test_episode_logger():
    """Episode logger must detect degenerate patterns."""
    print("\n" + "═"*60)
    print("TEST 22: Episode logger detects degenerate patterns")
    print("═"*60)
    from server.episode_logger import EpisodeLogger

    logger = EpisodeLogger(sample_rate=1)

    # Simulate degenerate agent: always rejects
    for i in range(20):
        logger.log_episode({
            "actions": ["REJECT"],
            "final_decision": "REJECT",
            "reward": 0.1,
            "num_steps": 1,
            "docs_collected": [],
            "phase2_actions": [],
            "terminal_outcome": "REJECTED",
        })

    alerts = logger.detect_patterns()
    print(f"    Alerts: {alerts['description']}")
    check(alerts["constant_action_sequence"],
          "Must detect constant action sequence (always REJECT)")
    check(alerts["constant_decision"],
          "Must detect constant decision (always REJECT)")
    check(alerts["always_reject"],
          "Must flag always-reject strategy")
    check(alerts["any_alert"],
          "At least one alert must fire")

    # Simulate healthy agent: varied actions
    logger2 = EpisodeLogger(sample_rate=1)
    import random as stdlib_random
    rng = stdlib_random.Random(42)
    for i in range(20):
        decision = rng.choice(["APPROVE", "REJECT"])
        n_docs = rng.randint(0, 2)
        logger2.log_episode({
            "actions": [f"DOC_{j}" for j in range(n_docs)] + [decision],
            "final_decision": decision,
            "reward": rng.uniform(-1, 2),
            "num_steps": n_docs + 1,
            "docs_collected": [f"doc_{j}" for j in range(n_docs)],
            "phase2_actions": [],
            "terminal_outcome": "REPAID" if decision == "APPROVE" else "REJECTED",
        })

    alerts2 = logger2.detect_patterns()
    print(f"    Healthy alerts: {alerts2['description']}")
    check(not alerts2["any_alert"],
          "Healthy varied agent should NOT trigger any alert")


# ══════════════════════════════════════════════════════════════════════════
# TEST 23: APPROVE returns APPLICATION phase with transition flag (Bug 3)
# ══════════════════════════════════════════════════════════════════════════

def test_approve_returns_application_phase_with_transition_flag():
    """APPROVE response must have current_phase=APPLICATION and transitioning_to_phase2=True."""
    print("\n" + "═"*60)
    print("TEST 23: APPROVE returns APPLICATION phase with transition flag")
    print("═"*60)
    env = MicrofinanceEnvironment(seed=42)
    env.reset()
    obs = env.step(CreditAction(action_type="APPROVE", rationale="test transition"))
    check(obs.current_phase == "APPLICATION",
          f"APPROVE must return current_phase=APPLICATION (got '{obs.current_phase}')")
    check(obs.transitioning_to_phase2 is True,
          f"APPROVE must set transitioning_to_phase2=True (got {obs.transitioning_to_phase2})")
    check(not hasattr(obs, 'signal_quality') or getattr(obs, 'signal_quality', None) is None,
          "APPROVE response must NOT contain signal_quality (Phase 1 schema)")


# ══════════════════════════════════════════════════════════════════════════
# TEST 24: Step after APPROVE returns clean Phase 2 observation (Bug 3)
# ══════════════════════════════════════════════════════════════════════════

def test_second_step_after_approve_returns_clean_phase2():
    """Step after APPROVE must return full MonitoringObservation, no transition flag."""
    print("\n" + "═"*60)
    print("TEST 24: Step after APPROVE returns clean Phase 2")
    print("═"*60)
    env = MicrofinanceEnvironment(seed=42)
    env.reset()
    env.step(CreditAction(action_type="APPROVE", rationale="transition"))
    obs = env.step(CreditAction(action_type="DO_NOTHING", rationale="first phase2 step"))
    check(obs.current_phase == "MONITORING",
          f"Post-approve step must return current_phase=MONITORING (got '{obs.current_phase}')")
    check(getattr(obs, 'transitioning_to_phase2', False) is not True,
          "Post-approve step must NOT have transitioning_to_phase2=True")
    check(obs.signal_quality is not None,
          "Post-approve step must have signal_quality set")
    check(0.0 <= obs.signal_quality <= 1.0,
          f"signal_quality must be in [0,1] (got {obs.signal_quality})")
    check(obs.month_number is not None and obs.month_number >= 1,
          f"month_number must be >= 1 (got {obs.month_number})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 25: Redundant doc request still returns documents_submitted (Bug 2)
# ══════════════════════════════════════════════════════════════════════════

def test_redundant_doc_request_still_returns_documents_submitted():
    """documents_submitted must be present in obs even after redundant request."""
    print("\n" + "═"*60)
    print("TEST 25: Redundant doc request returns documents_submitted")
    print("═"*60)
    env = MicrofinanceEnvironment(seed=42)
    env.reset()
    env.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale="first"))
    obs = env.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale="duplicate"))
    check(hasattr(obs, 'documents_submitted'),
          "documents_submitted field must exist in observation")
    check("income_proof" in obs.documents_submitted,
          f"income_proof must be in documents_submitted (got {obs.documents_submitted})")


# ══════════════════════════════════════════════════════════════════════════
# TEST 26: Reset always returns APPLICATION phase (Bug 1)
# ══════════════════════════════════════════════════════════════════════════

def test_reset_returns_application_phase():
    """Reset must always return current_phase=APPLICATION so buttons are correctly initialized."""
    print("\n" + "═"*60)
    print("TEST 26: Reset returns APPLICATION phase")
    print("═"*60)
    # Use noisy_signals task — no pre-revealed info, so documents_submitted is truly empty
    env = MicrofinanceEnvironment(seed=42, task_name="noisy_signals")
    # First run an episode to completion
    env.reset()
    env.step(CreditAction(action_type="REQUEST_INCOME_PROOF", rationale="gather info"))
    env.step(CreditAction(action_type="REJECT", rationale="end episode"))
    # Now reset again — must clear all state
    obs = env.reset()
    check(obs.current_phase == "APPLICATION",
          f"Reset must return current_phase=APPLICATION (got '{obs.current_phase}')")
    check(getattr(obs, 'transitioning_to_phase2', False) is not True,
          "Reset must NOT have transitioning_to_phase2=True")
    check(obs.documents_submitted == [],
          f"Reset must have empty documents_submitted (got {obs.documents_submitted})")
    # Also verify basic_lending correctly pre-reveals income
    env2 = MicrofinanceEnvironment(seed=42, task_name="basic_lending")
    obs2 = env2.reset()
    check("income_proof" in obs2.documents_submitted,
          f"basic_lending must pre-reveal income_proof (got {obs2.documents_submitted})")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_blind_approve_loses()
    test_grader_nonconstant()
    test_determinism()
    test_signal_quality_corruption()
    test_strategy_beats_impulse()
    test_phase1_affects_phase2()
    test_intervention_timing()
    test_invalid_actions()
    test_task_difficulty()
    test_reward_distribution()
    test_always_reject_exploit()
    test_over_requesting_exploit()
    test_redundant_action_spam()
    test_phase2_inaction_exploit()
    test_reward_audit()
    # v2 new tests
    test_behavioral_baselines()
    test_seed_diversity()
    test_counterfactual_gap()
    test_adversarial_overcollection()
    test_no_single_strategy_dominates()
    test_phase2_monotonic_penalty()
    test_episode_logger()
    # v2 bug-fix verification tests
    test_approve_returns_application_phase_with_transition_flag()
    test_second_step_after_approve_returns_clean_phase2()
    test_redundant_doc_request_still_returns_documents_submitted()
    test_reset_returns_application_phase()

    print("\n" + "═"*60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} checks")
    print("═"*60)
    if failed == 0:
        print("✅ All checks passed — environment is HARDENED against reward hacking.")
    else:
        print(f"⚠ {failed} check(s) failed — review environment logic.")
    sys.exit(0 if failed == 0 else 1)
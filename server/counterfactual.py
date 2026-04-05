"""
counterfactual.py — Soft counterfactual oracle for the Microfinance environment.

Design Philosophy
═════════════════
The counterfactual module answers: "What would a perfectly-informed agent have done?"

This is used as a SOFT PENALTY (10–15%), not a hard override:
  - If agent decision matches optimal → no penalty
  - If agent had HIGH confidence but chose wrong → moderate penalty
  - If agent had LOW confidence and chose wrong → small or no penalty (exploration OK)

The key insight: we only penalize counterfactual mismatches when the agent
HAD ENOUGH INFORMATION to know better. This avoids punishing exploration.
"""

from __future__ import annotations


def compute_optimal_decision(
    ground_truth: str,
    default_prob: float,
    is_borderline: bool = False,
) -> str:
    """
    What a perfectly informed agent would do.

    Returns "APPROVE" or "REJECT" based on the hidden ground truth.
    For borderline cases (prob near 0.50), the "correct" decision is ambiguous.
    """
    return ground_truth  # The oracle knows the truth


def counterfactual_gap(
    agent_decision: str,
    agent_confidence: float,
    ground_truth: str,
    default_prob: float,
    is_borderline: bool = False,
) -> float:
    """
    Returns a soft penalty in [0.0, 1.0] based on how far from optimal the
    agent's decision was AND how much information it had.

    Design:
      - If agent matches optimal: 0.0 (no gap)
      - If agent is wrong AND had high confidence: up to 1.0 (should have known)
      - If agent is wrong AND had low confidence: 0.1–0.3 (understandable)
      - Borderline cases reduce the penalty (genuine ambiguity)

    The gap is intended to be multiplied by a small weight (0.10–0.15) in the grader.
    """
    optimal = compute_optimal_decision(ground_truth, default_prob, is_borderline)

    if agent_decision == optimal:
        return 0.0  # No gap — decision matches oracle

    # Agent chose wrong — scale penalty by confidence
    # High confidence + wrong = agent gathered info but still got it wrong
    # Low confidence + wrong = agent didn't investigate enough (or case was hard)

    if is_borderline:
        # Borderline cases: the "right" answer is genuinely ambiguous
        # Penalty is soft — even an oracle would be uncertain
        ambiguity_discount = 0.5
    else:
        # Clear case: if agent had info and still chose wrong, that's worse
        ambiguity_discount = 1.0

    if agent_confidence >= 0.65:
        # Had sufficient info but made wrong call — moderate penalty
        base_gap = 0.7
    elif agent_confidence >= 0.35:
        # Partial info — somewhat understandable
        base_gap = 0.4
    else:
        # Zero or minimal info — agent didn't bother to investigate
        # Still penalized, but lightly (can't know what you don't look at)
        base_gap = 0.25

    return round(base_gap * ambiguity_discount, 3)


def counterfactual_grade_modifier(
    agent_decision: str,
    agent_confidence: float,
    ground_truth: str,
    default_prob: float,
    is_borderline: bool = False,
) -> float:
    """
    Returns a score modifier in [0.0, 1.0] for the grader.
    
    1.0 = perfect (no counterfactual gap)
    0.0 = worst case (high-confidence wrong decision on clear case)
    
    Intended to be used as a soft component (10–15% weight) in the composite score.
    """
    gap = counterfactual_gap(
        agent_decision, agent_confidence,
        ground_truth, default_prob, is_borderline,
    )
    return round(1.0 - gap, 3)

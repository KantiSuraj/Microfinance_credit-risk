"""
data_generator.py — Synthetic applicant dataset with deliberate noise,
conflicting signals, and borderline cases.

Design philosophy
-----------------
A trivial dataset (income ∝ creditworthiness) would let the agent learn a
one-feature lookup table.  We deliberately break that by injecting:

  1. CONFLICTING SIGNALS   — high income + many dependents + past default
  2. BORDERLINE CASES      — marginal income, one old default, good recent streak
  3. NOISY LABELS          — ~8 % of cases have a label that contradicts the
                             naive signal (real-world messiness)
  4. OCCUPATION TRAPS      — gig/seasonal workers with high incomes that are
                             structurally unstable
  5. REGION ASYMMETRY      — rural applicants get higher loan-to-income ratios
                             as a proxy for informal economy norms

The ground_truth_label is derived from a scoring function that aggregates
multiple factors with randomised weights, then perturbed with noise.
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Optional, List
import uuid


# ── Occupation registry ────────────────────────────────────────────────────

STABLE_JOBS   = ["salaried_govt", "salaried_private", "teacher", "nurse"]
VARIABLE_JOBS = ["small_trader", "contractor", "freelancer", "artisan"]
SEASONAL_JOBS = ["farmer", "fisherman", "construction_daily", "street_vendor"]

REGION_TIERS  = ["rural", "semi_urban", "urban"]

SENIOR_COMMENTS = {
    # keyed by rough risk band
    "low_risk":   [
        "Strong repayment history. Income verified. Recommend approval.",
        "Consistent SHG participation. Low default risk.",
        "Income stable for 18+ months. Dependents manageable.",
    ],
    "medium_risk": [
        "One past default but streak of 12 months. Monitor closely.",
        "Income seasonal — suggest smaller tranche with review at 6 months.",
        "Borderline. Group guarantee may suffice.",
    ],
    "high_risk": [
        "Two defaults in last 3 years. Needs co-applicant.",
        "Income source unstable. Loan-to-income ratio too high.",
        "Multiple active loans detected. Overleveraged.",
    ],
}


# ── Core applicant record ──────────────────────────────────────────────────

@dataclass
class ApplicantProfile:
    applicant_id: str
    dependents: int
    occupation: str
    income_source_stability: str       # stable / variable / seasonal
    monthly_income: float
    loan_amount_requested: float
    region_tier: str
    previous_loans: int
    past_defaults: int
    repayment_streak: int              # consecutive on-time months

    # Ground truth (computed, not directly observable)
    ground_truth_label: str            # "APPROVE" | "REJECT"
    true_default_probability: float
    risk_band: str                     # low / medium / high (for senior comment)

    # Noise flags (for internal audit / grader explanation)
    has_conflicting_signal: bool = False
    is_borderline: bool = False
    has_noisy_label: bool = False


def _score_applicant(
    income: float,
    loan_ask: float,
    dependents: int,
    stability: str,
    past_defaults: int,
    repayment_streak: int,
    previous_loans: int,
    region: str,
    rng: random.Random,
) -> tuple[float, str]:
    """
    Returns (default_probability, risk_band).
    Score is a weighted sum — weights are randomised slightly per call
    to simulate real-world model variance.
    """
    score = 0.0

    # --- Debt-to-income ratio (most predictive, but not always) -----------
    lti = loan_ask / max(income, 1)
    # higher LTI → worse
    score += rng.uniform(0.8, 1.2) * min(lti / 3.0, 1.0) * 30

    # --- Dependents burden ------------------------------------------------
    dep_burden = dependents / max(income / 5000, 1)
    score += rng.uniform(0.7, 1.3) * min(dep_burden, 1.0) * 20

    # --- Employment stability ----------------------------------------------
    stability_penalty = {"stable": 0, "variable": 12, "seasonal": 22}
    score += rng.uniform(0.9, 1.1) * stability_penalty[stability]

    # --- Credit history ----------------------------------------------------
    if past_defaults > 0:
        # Recency matters — offset by repayment streak
        recency_discount = max(0, 1 - repayment_streak / 18)
        score += rng.uniform(0.8, 1.2) * past_defaults * 18 * recency_discount

    if previous_loans > 0 and past_defaults == 0:
        # Good track record → reduce score
        score -= rng.uniform(0.8, 1.2) * min(repayment_streak / 6, 3) * 5

    # --- Region adjustment ------------------------------------------------
    # Rural lending is common in microfinance; relax threshold slightly
    if region == "rural":
        score -= rng.uniform(2, 6)

    # Clamp to [0, 100] then convert to probability
    score = max(0.0, min(score, 100.0))
    prob  = score / 100.0

    if prob < 0.33:
        band = "low_risk"
    elif prob < 0.60:
        band = "medium_risk"
    else:
        band = "high_risk"

    return prob, band


def generate_applicant(
    rng: Optional[random.Random] = None,
    force_conflict: bool = False,
    force_borderline: bool = False,
) -> ApplicantProfile:
    """
    Generate one synthetic applicant with optional trait forcing.
    """
    if rng is None:
        rng = random.Random()

    applicant_id = str(uuid.uuid4())[:8]
    region = rng.choice(REGION_TIERS)

    # ── Occupation & stability ────────────────────────────────────────────
    occupation_pool = STABLE_JOBS + VARIABLE_JOBS + SEASONAL_JOBS
    occupation = rng.choice(occupation_pool)
    if occupation in STABLE_JOBS:
        stability = "stable"
    elif occupation in VARIABLE_JOBS:
        stability = "variable"
    else:
        stability = "seasonal"

    # ── Income (log-normal centred near ₹18k, heavy tail) ─────────────────
    log_mean  = math.log(18_000)
    log_sigma = 0.65
    income    = rng.lognormvariate(log_mean, log_sigma)
    income    = round(max(4_000, min(income, 120_000)), -2)  # snap to hundreds

    # ── Loan amount ───────────────────────────────────────────────────────
    # Typically 2–6× monthly income for micro-loans
    lti_multiplier = rng.uniform(1.5, 6.5)
    loan_ask       = round(income * lti_multiplier, -3)  # snap to thousands

    # ── Dependents ────────────────────────────────────────────────────────
    dependents = rng.choices(
        [0, 1, 2, 3, 4, 5, 6],
        weights=[5, 15, 25, 25, 15, 10, 5]
    )[0]

    # ── Credit history ─────────────────────────────────────────────────────
    previous_loans = rng.randint(0, 5)
    past_defaults  = 0
    if previous_loans > 0:
        # ~30 % chance of at least one default
        past_defaults = rng.choices(
            range(0, min(previous_loans + 1, 4)),
            weights=[60, 25, 10, 5][: min(previous_loans + 1, 4)]
        )[0]

    repayment_streak = 0
    if previous_loans > 0:
        repayment_streak = rng.randint(0, 30)

    # ── Inject conflicting signal ─────────────────────────────────────────
    has_conflict = False
    if force_conflict or rng.random() < 0.20:
        has_conflict = True
        if rng.random() < 0.5:
            # High income but many dependents + a past default
            income    = rng.uniform(45_000, 90_000)
            dependents = rng.randint(4, 6)
            past_defaults = 1
            repayment_streak = rng.randint(0, 8)
        else:
            # Low income but spotless history + stable job
            income    = rng.uniform(6_000, 14_000)
            occupation = rng.choice(STABLE_JOBS)
            stability  = "stable"
            past_defaults = 0
            repayment_streak = rng.randint(18, 36)

    # ── Inject borderline case ────────────────────────────────────────────
    is_borderline = False
    if force_borderline or rng.random() < 0.18:
        is_borderline = True
        income    = rng.uniform(14_000, 22_000)
        loan_ask  = income * rng.uniform(2.8, 3.5)
        past_defaults = 1
        repayment_streak = rng.randint(10, 16)

    # ── Ground truth scoring ──────────────────────────────────────────────
    default_prob, risk_band = _score_applicant(
        income, loan_ask, dependents, stability,
        past_defaults, repayment_streak, previous_loans, region, rng
    )

    # ── Noisy label (~8 % flip) ───────────────────────────────────────────
    has_noise = False
    if rng.random() < 0.08:
        has_noise     = True
        default_prob  = 1.0 - default_prob  # flip probability

    label = "REJECT" if default_prob > 0.50 else "APPROVE"

    return ApplicantProfile(
        applicant_id          = applicant_id,
        dependents            = dependents,
        occupation            = occupation,
        income_source_stability = stability,
        monthly_income        = income,
        loan_amount_requested = loan_ask,
        region_tier           = region,
        previous_loans        = previous_loans,
        past_defaults         = past_defaults,
        repayment_streak      = repayment_streak,
        ground_truth_label    = label,
        true_default_probability = default_prob,
        risk_band             = risk_band,
        has_conflicting_signal= has_conflict,
        is_borderline         = is_borderline,
        has_noisy_label       = has_noise,
    )


def generate_dataset(
    n: int = 300,
    seed: int = 42,
    conflict_ratio: float = 0.20,
    borderline_ratio: float = 0.18,
) -> list[ApplicantProfile]:
    """
    Generate a fixed dataset. Guarantees a minimum proportion of
    conflicting and borderline cases regardless of random draws.
    """
    rng      = random.Random(seed)
    profiles = []

    n_conflict    = int(n * conflict_ratio)
    n_borderline  = int(n * borderline_ratio)
    n_rest        = n - n_conflict - n_borderline

    for _ in range(n_conflict):
        profiles.append(generate_applicant(rng, force_conflict=True))
    for _ in range(n_borderline):
        profiles.append(generate_applicant(rng, force_borderline=True))
    for _ in range(n_rest):
        profiles.append(generate_applicant(rng))

    rng.shuffle(profiles)
    return profiles
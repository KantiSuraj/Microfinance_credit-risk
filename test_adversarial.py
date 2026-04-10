"""Quick test: do adversarial profiles genuinely score as APPROVE?"""
import random
from server.data_generator import generate_applicant

rng = random.Random(256)
approve_count = 0
for i in range(10):
    p = generate_applicant(rng, force_adversarial=True)
    label = p.ground_truth_label
    prob = p.true_default_probability
    if label == "APPROVE":
        approve_count += 1
    lti = p.loan_amount_requested / p.monthly_income
    print(f"  [{i+1}] label={label} prob={prob:.3f} "
          f"income={p.monthly_income:.0f} lti={lti:.2f} "
          f"occ={p.occupation} streak={p.repayment_streak}")

print(f"\nAPPROVE rate: {approve_count}/10 ({approve_count*10}%)")

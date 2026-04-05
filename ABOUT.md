### 1. Sequential Information Gathering & Two-Phase Lifecycle

- **Standard ML:** A one-shot binary decision based on a complete static row of data.
- **RL Microfinance:** A two-phase loan lifecycle forcing progressive decisions.
    - **Phase 1 (Application):** Agent is presented with partial applicant data and must navigate discovery (e.g., requesting income proof or credit history) to decide whether to APPROVE or REJECT.
    - **Phase 2 (Monitoring):** For approvals, the agent monitors exactly 12 months of repayments. The agent must make monthly intervention decisions given noisy observations.
- Multi-step reasoning across contrasting action spaces fundamentally bridges the gap from a static classifier to dynamic portfolio management.

### 2. A Richer, Progressive Action Space

Instead of merely a classifier, the agent must orchestrate policies involving multiple interacting actions across phases:
- **Phase 1 Actions:** `APPROVE`, `REJECT` (Terminal), `REQUEST_INCOME_PROOF`, `REQUEST_CREDIT_HISTORY` (Information Gathering), `FLAG_FOR_REVIEW` (Escalation).
- **Phase 2 Actions:** `DO_NOTHING`, `SEND_REMINDER`, `RESTRUCTURE_LOAN`, `ESCALATE_TO_RECOVERY`.

### 3. Trade-offs, Asymmetric Cost Modeling & Signal Quality

The environment explicitly models real-world business and societal trade-offs. Asking for documents costs time and resources. Business penalties are asymmetric across the whole lifecycle:
- **Phase 1 Costs:** `-0.10` per document request, `-0.15` for flagging for senior reviews.
- **Phase 1 Outcomes:** Correct Approve (+1.0), Wrong Approve (-2.0), Correct Reject (+0.6), Wrong Reject (-1.0), Indecision / Timeout (-1.5).
- **Phase 2 Interventions:** Sending a Reminder costs `-0.05` but reduces default risk by 5%. Restructuring a loan costs `-0.20` but heavily reduces risk by 20%. Extricating a failing loan (Escalate) costs `-0.50`.
- **Phase 2 Terminal:** Full repayment yields a +1.5 bonus on top of Phase 1; Default penalties immediately wipe out Phase 1 profit domains (-2.5 terminal penalty).

**The Signal Quality Propagation (Flagship Mechanic):**
When Phase 1 ends, Phase 2 signal quality is explicitly computed from what documents the agent collected. Did the agent gather everything? Signal Quality = 90%. Did they rush the Phase 1 approval with no documents? Signal Quality = 60%. This value flips each month's true payment observation with probability $1 - \text{signal\_quality}$. Therefore, an agent that rushed approval now spends 12 months making crucial intervention decisions using corrupted, unreliable signals.

---

## Code Structure & Key Components

### 1. State Management & Discovery (`models.py` & `client.py`)
- **Two-Phase Routing:** `client.py` transparently parses outputs from `microfinance_env_environment.py`, returning either an `ApplicantObservation` or `MonitoringObservation` dependent on the current phase.
- **Missing Information:** Phase 1 features `monthly_income`, `repayment_streak`, etc., are explicitly hidden (`None`) until explicitly gathered by actions. Phase 2 strictly hides the true Default Probability metrics while exposing `missed_streak` and cumulative performance.
- **Strict Budgeting:** The agent operates under a hard `max_steps = 7` strict limit for Phase 1 and exactly 12 total months for Phase 2.

### 2. Synthetic Data Generation (`server/data_generator.py`)
A dataset generator that accurately models real-world socioeconomic turbulence:
- **Income & Occupation:** Log-normal distributions around emerging market incomes, grouped by job stabilities (stable vs. variable/seasonal). Some high-income borrowers feature seasonal jobs, elevating actual default risk despite good raw numbers.
- **Noise & Borderline Distribution:** 
  - **Conflicting Signals:** High income but poor repayment history, or very low income but flawless repayment streaks.
  - **Label Noise:** Deliberate paradoxes injected to prevent pure over-fitting.

### 3. Trajectory-Aware Evaluation Logic (`server/grader.py`)
Grading an RL agent purely on binary accuracy entirely dismisses the purpose of Phase 2. Thus, the programmatic grader utilizes a trajectory-aware reward system spanning four continuous dimensions:
- **Phase 1 Score (35%):** Was the approval/rejection decision sound given evidence, and was it efficiently gathered?
- **Phase 2 Score (35%):** Were monthly observations interpreted correctly? Did it identify risk and react appropriately?
- **Timing Score (20%):** **The key metric impossible for a classifier.** Intervening at Month 3 before a streak of missed payments scores 0.85, whereas intervening at Month 8 on a completely deteriorated loan scores 0.2. Late actions are fundamentally too weak to matter.
- **Information Flow Score (10%):** Recognizing that higher Phase 1 Signal Quality translates to superior Phase 2 monitoring insights.

#### LLM Qualitative Grader
In addition to the programmatic baseline, `grader.py` features an LLM harness to qualitatively score agent trajectories 0–10 based on:
1. **Phase 1 Reasoning:** Did the agent request precisely the right evidence?
2. **Phase 2 Management:** Did it interpret noise effectively and deploy optimal intervention typologies?
3. **Trajectory Coherence:** Is there evidence the agent traded off Phase 1 costs against Phase 2 observational quality?
4. **Financial Inclusion:** Did the agent avoid over-rejecting based on raw income alone?


python test_env.py. Here's what each test validates:

| Test                     | What It Attacks                                   | Why It Matters                     |
|--------------------------|--------------------------------------------------|------------------------------------|
| 1. Blind approve loses   | Can the agent always approve and win?            | Anti-exploit                       |
| 2. Grader non-constant   | Does grader always return the same score?        | Disqualification check             |
| 3. Determinism           | Same seed = same result?                         | Reproducibility requirement        |
| 4. Signal quality        | Does the noise mechanic actually work?           | Core design claim                  |
| 5. Strategy > impulse    | Does gathering docs actually help?               | RL signal validity                 |
| 6. Phase 1→2 causality   | Do Phase 1 choices affect Phase 2?               | Two-phase claim                    |
| 7. Intervention timing   | Does reactive beat passive?                      | Sequential reasoning               |
| 8. Invalid actions       | Wrong-phase actions crash?                       | Robustness                         |
| 9. Task difficulty       | Easy > Hard scores?                             | Structural variation               |
| 10. Score distribution   | Scores well-spread [0, 1]?                       | Meaningful reward signal           |
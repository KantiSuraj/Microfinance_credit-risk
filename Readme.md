# Microfinance Credit Decision Environment

This repository contains a robust Reinforcement Learning (RL) environment designed for simulating and solving microfinance credit decisions. Unlike traditional machine learning approaches that predict a binary outcome from a static dataset, this environment captures the **sequential, cost-aware reality of loan under-writing** by forcing an agent to actively gather intelligence under tight efficiency constraints.

---

## The Paradigm Shift: RL vs Traditional ML

Traditional ML models for credit risk assume all data is neatly available upfront. They have no notion of asking for more information, delaying decisions, or factoring in the operational cost of prolonged analysis. This RL version introduces critical paradigm shifts:

### 1. Sequential Information Gathering
- **Standard ML:** A one-shot binary decision based on a complete static row of data.
- **RL Microfinance:** A sequence of progressive decisions. For example:
     - **Step 1:** Data is initially incomplete (only occupation/dependents visible).
     - **Step 2:** Agent requests income proof (incurring a time/step penalty).
     - **Step 3:** Agent requests credit history (incurring another penalty).
     - **Step 4:** New info proves satisfactory → Agent Approves.
- Multi-step reasoning and dynamic state discovery is impossible in standard supervised ML but acts as the fundamental loop here.

### 2. A Richer, Progressive Action Space
Instead of merely a classifier, the agent must develop a policy involving multiple interacting actions:
- `APPROVE` / `REJECT` (Terminal Actions)
- `REQUEST_INCOME_PROOF` / `REQUEST_CREDIT_HISTORY` (Information Actions)
- `FLAG_FOR_REVIEW` (Escalation Action)

### 3. Trade-offs & Asymmetric Cost Modeling
The environment explicitly models real-world business and societal trade-offs. Asking for documents costs time and resources. Even more critically, the business penalties are deeply asymmetric:
- **Correct Approve (+1.0):** Ideal scenario.
- **Wrong Approve (-2.0):** Massive penalty for issuing bad debt (false positive).
- **Correct Reject (+0.6):** Good risk management, but doesn't grow the business like a good loan.
- **Wrong Reject (-1.0):** Opportunity loss and potential harm to creditworthy borrowers (false negative).
- **Indecision (-1.5):** Failing to make a decision within 7 steps forces an auto-rejection.
- **Operational Costs:** `-0.1` per document request, `-0.15` for flagging for senior reviews.

RL optimizes the "best decision under these constraints," capturing lending dynamics that ML fails to encompass.

---

## Code Structure & Key Components

### 1. State Management & Discovery (`Models.py` & `Microfinance enviroment.py`)
At the start of an episode, critical fields like `monthly_income`, `income_source_stability`, `past_defaults`, `repayment_streak`, and `senior_review_comment` are explicitly hidden (`None`).
- **Active Discovery:** Each document action fetches specific fields and updates the `ApplicantObservation` state.
- **Strict Budgeting:** The agent operates under a hard `max_steps = 7` strict limit. 
- **No Free Lunches:** Requesting the same document twice yields no new info but still incurs a wasted step penalty. The agent must memorize its progress to stay financially and computationally efficient.

### 2. Synthetic Data Generation (`Data generator.py`)
A trivial dataset where income purely correlates with creditworthiness would allow an agent to succeed with a lazy, one-dimensional threshold. This generator accurately models real-world socioeconomic turbulence:
- **Income Distribution:** Uses a log-normal distribution centered around ₹18,000 with a heavy tail, accurately mirroring real emerging markets.
- **Occupation Traps:** Segregates occupations into `STABLE_JOBS` (government, teachers), `VARIABLE_JOBS` (freelancers, artisans), and `SEASONAL_JOBS` (farmers, daily construction). An applicant with high income but seasonal stability often faces higher default risk.
- **Nuanced Ground Truth Scoring:** Rather than a simple cut-off, ground truth default probability calculates debt-to-income limits, dependent burdens, stability penalties, and recent repayment streaks mitigating older defaults. Furthermore, rural lending logic acts as a relaxing modifier to mimic informal economy practices.
- **Noise & Borderline Distribution:** 
  - **20% Conflicting Signals:** High income but multiple dependents and a recent default; or very low income but flawless repayment streaks.
  - **18% Borderline Cases:** Marginal LTI ratios relying entirely on a good recent sprint of repayments.
  - **8% Label Noise:** Deliberate paradoxes injected to prevent pure over-fitting.
- Outcome ratios safely distribute as ~54% Approve vs 46% Reject.

### 3. Comprehensive Evaluation Logic (`Grader.py`)
Grading an RL agent purely on binary accuracy misses qualitative mastery. Thus, the programmatic grader utilizes a tripartite reward system:
- **Correctness (50%):** Does the decision match ground truth? If an agent gets it wrong on an objectively ambiguous case, the grader awards a partial ambiguity credit (up to 0.3) instead of a pure zero.
- **Efficiency (30%):** Tracks steps taken vs. an ideal baseline. (e.g., An easy case should take ≤3 steps; borderline cases permit ≤5 steps).
- **Information Strategy (20%):** Rewards gathering both income and credit documents on ambiguous cases, while penalizing cautious agents that redundantly request documents for overwhelmingly easy cases.

#### Verified Grader Outcomes:
- `1.000` = Correct outcome + lean strategic execution.
- `1.000` = Impulsive correct (Succeeds via pure luck on an easy case).
- `0.635` = Wrong choice but heavily conflicting profile (partial credit salvaged).
- `0.560` = Impulsive wrong (Immediate failure).
- `0.200` = Total indecisiveness (Times out).

#### LLM Qualitative Grader (Round 2)
In addition to the programmatic baseline, `Grader.py` provides an LLM harness (`build_llm_grader_prompt`) to critique agent traces qualitatively scoring 0–10 along three vital dimensions:
1. **Reasoning Quality:** Did the agent identify conflicting signals naturally?
2. **Decision Accuracy:** Did it gather enough evidence exactly when borderline tension demanded it?
3. **Financial Inclusion:** Did it factor in the applicant's community tier and avoid rejecting based exclusively on raw income?

---
title: Microfinance Credit Decision Environment
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---
# 🏦 Microfinance Credit Decision Environment

> An RL environment where an agent must decide whether to approve a micro-loan — but it can't see the full picture. It must pay to gather evidence, and every wrong approval costs lives.

---

## ❓ The Problem: Decisions Under Uncertainty

In India, **80 million+ microfinance borrowers** depend on loan officers making fast decisions with **incomplete, noisy, and conflicting information**.

A real loan officer faces this every day:

> *"This applicant earns ₹18,000/month (good) but has 2 past defaults (bad). Do I approve? Or do I spend time and money pulling their credit history first?"*

This is **not** a classification problem.  
This is a **sequential decision-making** problem under uncertainty.

---

## 🧠 Why Reinforcement Learning?

A classifier sees all the data and makes one prediction. **That's not how lending works.**

| | Traditional ML | This RL Environment |
|---|---|---|
| **Input** | Complete feature row | Partial, hidden fields |
| **Decision** | One-shot predict | Multi-step: gather → analyze → decide |
| **Cost model** | None | Every action has a cost |
| **Consequence** | Accuracy metric | Wrong approval → 12 months of default risk |
| **Time horizon** | Instant | Up to 19 steps across 2 phases |

**The agent must reason sequentially** because:
1. Information is **hidden** until explicitly requested (and each request costs money)
2. The quality of Phase 1 investigation **directly determines** how noisy Phase 2 observations will be
3. A rushed approval means **12 months of monitoring with corrupted signals**

---

## 🔥 What Makes This Hard?

The agent faces a **three-way trade-off** at every step:

```
                    ┌─────────────────┐
                    │   INFORMATION   │
                    │   (cost of      │
                    │    documents)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼───────┐      │    ┌─────────▼───────┐
    │      RISK       │      │    │      DELAY      │
    │  (default       │◄─────┘───►│  (step          │
    │   penalty)      │           │   penalties)    │
    └─────────────────┘           └─────────────────┘
```

- **Gather too little** → approve blindly → borrower defaults → **-2.50 penalty**
- **Gather too much** → escalating step costs eat your reward → **diminishing returns**
- **Wait too long to intervene** → default becomes inevitable → **too late to save**

No single strategy wins. The agent must **adapt to each applicant**.

---

## 🎬 Demo: Dumb Agent vs. Smart Agent

This is the moment that shows **why RL matters**.

### ❌ Case 1: The Blind Agent (fails)

```
╔══════════════════════════════════════════════════════════════════╗
║  EPISODE START — Applicant #A-7291                              ║
║  Visible: occupation=farmer, dependents=3, loan=₹45,000        ║
║  Hidden:  monthly_income=???, credit_history=???, defaults=???  ║
║  Confidence: 0%                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Step 1: Agent → APPROVE  (no investigation)                     ║
║                                                                  ║
║  ⚠ Ground truth: this borrower has 3 past defaults               ║
║  ⚠ True default probability: 72%                                 ║
║  ⚠ Signal quality set to: 60% (worst — no docs collected)        ║
║                                                                  ║
║  Phase 1 Reward: -0.40 (blind approval penalty)                  ║
║                                                                  ║
║  ── Phase 2: Monitoring (12 months of corrupted signals) ──      ║
║                                                                  ║
║  Month 1: Observed ON_TIME  (actually MISSED — signal flipped!)  ║
║  Month 2: Observed ON_TIME  (actually MISSED — signal flipped!)  ║
║  Month 3: Observed MISSED   (actually MISSED — finally true)     ║
║  Month 4: Observed MISSED   (cumulative misses = 4)              ║
║                                                                  ║
║  ✖ LOAN DEFAULT — Borrower failed after 4 missed payments        ║
║                                                                  ║
║  ╔═══════════════════════════════════════╗                        ║
║  ║  TOTAL REWARD: -2.50                  ║                        ║
║  ║  GRADER SCORE: 0.12 / 1.00  ✖ FAIL   ║                        ║
║  ╚═══════════════════════════════════════╝                        ║
╚══════════════════════════════════════════════════════════════════╝
```

**What went wrong:** The agent approved without investigation. Phase 2 signals were 60% noisy — it literally couldn't tell missed payments from on-time ones. By the time real misses showed up, it was too late.

---

### ✅ Case 2: The Strategic Agent (succeeds)

```
╔══════════════════════════════════════════════════════════════════╗
║  EPISODE START — Same Applicant #A-7291                         ║
║  Visible: occupation=farmer, dependents=3, loan=₹45,000        ║
║  Hidden:  monthly_income=???, credit_history=???, defaults=???  ║
║  Confidence: 0%                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Step 1: Agent → REQUEST_CREDIT_HISTORY                          ║
║          📋 Revealed: 4 prior loans, 3 defaults, 0-month streak  ║
║          Confidence: 65%     Cost: -0.12                         ║
║                                                                  ║
║  Step 2: Agent → REJECT                                          ║
║          Rationale: "3 past defaults with 0 repayment streak     ║
║          indicates extreme risk. Rejection warranted."           ║
║                                                                  ║
║  ✓ Ground truth: REJECT — correct decision                       ║
║                                                                  ║
║  ╔═══════════════════════════════════════╗                        ║
║  ║  TOTAL REWARD: +0.53                  ║                        ║
║  ║  GRADER SCORE: 0.78 / 1.00  ✓ PASS   ║                        ║
║  ╚═══════════════════════════════════════╝                        ║
╚══════════════════════════════════════════════════════════════════╝
```

**What went right:** The agent spent **one step** (cost: -0.12) to reveal the most important signal — credit history. It saw 3 past defaults and immediately rejected. Clean, fast, informed.

> **The difference: +3.03 reward.** One document request changed everything.

---

## 🏗️ Two-Phase Lifecycle

The environment models the **full loan lifecycle**, not just the approval moment.

```
┌─────────────────────────────────────────────────────────┐
│                    PHASE 1: APPLICATION                  │
│                    (up to 7 steps)                        │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │ Observe  │───►│ Request  │───►│ Decide   │            │
│  │ partial  │    │ docs     │    │ approve/ │            │
│  │ info     │    │ (-cost)  │    │ reject   │            │
│  └──────────┘    └──────────┘    └─────┬────┘            │
│                                        │                 │
│  Actions: APPROVE | REJECT | REQUEST_INCOME_PROOF        │
│           REQUEST_CREDIT_HISTORY | FLAG_FOR_REVIEW       │
└────────────────────────────────────────┼─────────────────┘
                                         │
                    ┌────────────────────▼──────────────────┐
                    │   Signal Quality = f(docs collected)   │
                    │   More docs → clearer Phase 2 signals  │
                    │   No docs  → noisy, unreliable obs     │
                    └────────────────────┬──────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────┐
│                    PHASE 2: MONITORING                    │
│                    (up to 12 months)                      │
│                                                           │
│  Each month:                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ Observe  │───►│ Assess   │───►│ Act or   │             │
│  │ payment  │    │ risk     │    │ wait     │             │
│  │ (noisy!) │    │ trend    │    │          │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│                                                           │
│  Actions: DO_NOTHING | SEND_REMINDER (-0.05, -5% risk)   │
│           RESTRUCTURE_LOAN (-0.20, -20% risk)             │
│           ESCALATE_TO_RECOVERY (-0.50, terminates)        │
│                                                           │
│  Ends: loan repaid (+1.50) or default (-2.50) or escalate │
└───────────────────────────────────────────────────────────┘
```

## 🔄 Phase Transition (Important Detail)

When the agent selects `APPROVE`, the environment performs a clean two-step transition:

```
Step N:   Agent sends APPROVE
          └─ Returns Phase 1 observation (with transitioning_to_phase2=True flag)
          └─ Environment internally switches to MONITORING state

Step N+1: First monitoring step
          └─ Returns Phase 2 MonitoringObservation (signal_quality set, month_number=1)
          └─ transitioning_to_phase2 flag is absent — full Phase 2 schema in effect
```

**Why this design?** It avoids mixing observation types within a single step, which ensures:

- ❌ No undefined fields (prevents NaN errors in agent code)
- ❌ No fabricated signals (e.g., fake payment status at month 0)
- ✅ Clean RL trajectories — each step has exactly one schema
- ✅ Consistent API contract — agents and UI can rely on `transitioning_to_phase2` as an explicit boundary signal

---

### 🔑 The Key Mechanic: Signal Quality Propagation

This is what makes the two phases **causally linked**:

| Docs Collected in Phase 1 | Signal Quality | What It Means |
|---|---|---|
| None | 60% | **40% of payment observations are flipped.** Agent is nearly blind. |
| Income only | 75% | Better, but credit history matters more. |
| Credit history | 90% | **Agent can trust what it sees.** Interventions are well-timed. |

A rushed Phase 1 doesn't just cost a bad approval — it **corrupts 12 months of future observations**.

---

## 🖥️ Interactive Dashboard

The environment comes with a **live web dashboard** showing:

| Feature | What It Shows |
|---|---|
| 📋 Applicant Profile | Partial → revealed info as docs are requested |
| ⚡ Action History | Every step, cost, and rationale |
| 📊 Reward Breakdown | Per-step reward, cumulative score, grader dimensions |
| 📈 Phase 2 Timeline | Monthly payments, interventions, risk evolution |
| 🎯 Signal Quality | Visual indicator of observation reliability |

> **Visual learning > reading code.** See the agent's behavior step by step.

---

## 🧪 Step-by-Step: How an Episode Works

```
1. START EPISODE
   └─ Agent sees: occupation, dependents, loan amount, region
   └─ Hidden: income, credit history, defaults, streak

2. GATHER INFORMATION (optional, costs money)
   └─ REQUEST_INCOME_PROOF  → reveals ₹ income + stability
   └─ REQUEST_CREDIT_HISTORY → reveals loans, defaults, streak
   └─ FLAG_FOR_REVIEW → senior officer comment (costs more)

3. MAKE DECISION
   └─ APPROVE → Phase 2 begins (signal quality set by docs)
   └─ REJECT  → episode ends (must be justified by evidence)

4. MONITOR MONTHLY (if approved)
   └─ See noisy payment observation each month
   └─ Choose: wait, remind, restructure, or escalate
   └─ Risk evolves: misses compound, on-time heals

5. EPISODE ENDS
   └─ Loan repaid (+1.50) → great outcome
   └─ Default (-2.50) → catastrophic 
   └─ Escalated (-0.50) → damage control
```

---

## 🎯 Three Difficulty Levels

| Task | Difficulty | What Makes It Hard | Agent Skill Tested |
|---|---|---|---|
| `basic_lending` | 🟢 Easy | Income pre-revealed, clear signals, 6-month Phase 2 | Basic approve/reject reasoning |
| `noisy_signals` | 🟡 Medium | Conflicting features, hidden key info, wrong-doc penalties | **Strategic information gathering** |
| `adversarial_portfolio` | 🔴 Hard | Borderline case, +12% base risk, 75% max signal quality, 3-miss threshold | **Long-horizon intervention timing** |

---

## 📦 Project Structure

```
microfinance_env/
├── inference.py                  # Baseline LLM agent (runs all 3 tasks)
├── models.py                     # Action, Observation, State contracts
├── client.py                     # Two-phase client wrapper
├── openenv.yaml                  # OpenEnv specification
│── Dockerfile                    # Container deployment
├── server/
│   ├── app.py                    # FastAPI server (OpenEnv interface)
│   ├── microfinance_env_environment.py  # Core environment logic
│   ├── reward_engine.py          # Anti-hack hardened reward system
│   ├── grader.py                 # 5-dimension trajectory grader
│   ├── counterfactual.py         # Soft counterfactual oracle
│   ├── data_generator.py         # Synthetic dataset generator
│   ├── episode_logger.py         # Pattern detection & logging
│   
│
├── test_env.py                   # 26 adversarial tests (68 checks)
├── test_anticlassification.py    # Causal impact validation (Phase 1 → Phase 2)
└── requirements.txt
```

The environment exposes both HTTP and WebSocket interfaces.
The UI uses HTTP endpoints for simplicity, while agent clients
can connect via WebSocket for persistent sessions

---

## 🤖 Running Inference (Judge / Evaluation)

The evaluation pipeline is fully automated via `inference.py`.

### Required Environment Variables

```bash
export HF_TOKEN=your_huggingface_token
export IMAGE_NAME=surajkanti/microfinance-env
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct  # or any OpenAI-compatible model
```

### Run

```bash
python inference.py
```

### Live Environment (no local setup needed)

The environment is already deployed and publicly accessible:

> **🤗 HuggingFace Space:** https://kantisuraj-microfinance-env.hf.space

Point `inference.py` at this URL to skip Docker entirely.

### Minimal Host Dependencies

```bash
pip install openai httpx
```

Alternatively: `pip install -r requirements.txt`

---

## 🚀 Quick Start (Local)

```bash
# 1. Activate environment
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

# 2. Start server
uvicorn server.app:app --reload
# → http://localhost:8000

# 3. Run baseline agent
python inference.py
```

### Example: Programmatic Usage

```python
import asyncio
from client import MicrofinanceEnv
from models import CreditAction

async def main():
    async with MicrofinanceEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        
        # Phase 1 — investigate before deciding
        result = await env.step(CreditAction(action_type="REQUEST_CREDIT_HISTORY"))
        # Now we can see: past defaults, repayment streak, prior loans
        
        if result.observation.past_defaults > 2:
            result = await env.step(CreditAction(
                action_type="REJECT",
                rationale="Too many past defaults"
            ))
        else:
            result = await env.step(CreditAction(action_type="APPROVE"))
            
            # Phase 2 — monitor monthly, intervene when needed
            for month in range(12):
                if result.done:
                    break
                if result.observation.missed_streak >= 2:
                    action = "RESTRUCTURE_LOAN"  # aggressive intervention
                elif result.observation.cumulative_misses > 0:
                    action = "SEND_REMINDER"      # gentle nudge
                else:
                    action = "DO_NOTHING"          # all good
                result = await env.step(CreditAction(action_type=action))

asyncio.run(main())
```

---

## 🛡️ Anti-Reward-Hacking (9 Strategies)

The environment is hardened against exploitation. No lazy or dumb strategy can consistently score well.

| Strategy | How It Works |
|---|---|
| **Multi-Objective Reward** | 5-dimension grader — can't optimize one dimension and ignore others |
| **Counterfactual Testing** | Penalizes decisions the agent "should have known better" about |
| **Redundancy Detection** | Escalating penalties for repeated useless actions |
| **Efficiency U-Curve** | Both rushing (too few steps) AND over-investigating (too many) are penalized |
| **Diversity Testing** | Verified across 100 seeds — no fixed strategy averages > 0.70 |
| **Episode Logging** | Automatic pattern detection flags degenerate strategies |
| **Independent Audit** | 8-flag validator catches exploit patterns |
| **Behavioral Baselines** | Strict ordering: informed > random > blind > always-reject |
| **Monotonic Strategy Penalty** | Doing the same Phase 2 action 4+ months straight → escalating cost |

**Test suite: 26 tests, 68 checks — all passing.** ✅

---

## 🐳 Deployment

```bash
# Docker (Dockerfile is at the project root)
docker build -t microfinance-env .
docker run -p 8000:8000 microfinance-env

# OpenEnv
openenv push
```

---

## 📌 Status

✅ **Version 2.1** — Anti-Reward-Hacking hardened. 68/68 adversarial checks passing.

---

<p align="center">
  <b>Built by Suraj & Team </b><br>
  <i>"This environment simulates how loan officers make decisions with incomplete information.<br>
  The agent must decide whether to request more documents or act early — balancing risk and cost."</i>
</p>
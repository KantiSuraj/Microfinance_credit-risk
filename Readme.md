# 🏦 Microfinance Credit Decision Environment (OpenEnv)

An interactive reinforcement learning (RL) environment simulating real-world microfinance loan decisions under uncertainty across a **two-phase loan lifecycle**.

This environment models how loan officers make decisions with incomplete applicant data, and how post-approval monitoring requires strategic interventions to prevent defaults.

---

## 🚀 Problem Statement

India has millions of microfinance borrowers. Loan officers face two distinct challenges:
1. **Application Phase**: Working with incomplete data to make risk vs opportunity judgments.
2. **Monitoring Phase**: Observing noisy repayment signals to identify deteriorating borrowers before they default.

This environment simulates that process as a **sequential, long-horizon decision-making problem**.

---

## 🧠 Key Idea

Unlike traditional ML (predict approve/reject in one step), this environment forces the agent to:
1. **Plan across two phases** with different action spaces.
2. **Trade off information gathering cost against future signal quality**: Choosing not to verify credit history saves step costs but yields noisy, unreliable payment observations for the next 12 months.
3. **Execute time-sensitive interventions**: A late reminder has less impact than an early restructure.

---

## ⚙️ Environment Design

The episode is divided into two phases.

### 🔹 Phase 1: Application
**Action Space**:
* `APPROVE` (Moves to Phase 2)
* `REJECT` (Terminates episode)
* `REQUEST_INCOME_PROOF` 
* `REQUEST_CREDIT_HISTORY`
* `FLAG_FOR_REVIEW`

**Mechanics**:
* Start with missing/incomplete fields.
* Document requests reveal hidden fields but incur step penalties.
* "Signal Quality" for Phase 2 is determined by the documents collected here.

---

### 🔹 Phase 2: Monitoring (12 Months)
**Action Space**:
* `DO_NOTHING`
* `SEND_REMINDER`
* `RESTRUCTURE_LOAN`
* `ESCALATE_TO_RECOVERY` (Terminates episode)

**Mechanics**:
* Monthly payment observations are noisy and dependent on Phase 1 "Signal Quality".
* Interventions modify the borrower's underlying true default probability.
* Streak accumulations (missed vs on-time payments) dynamically shift the risk band week to week.
* Terminates upon full repayment (12 months), cumulative default threshold, or manual escalation.

---

## 🎯 Tasks (3 Difficulty Levels)

| Task | Difficulty | What It Tests |
|------|-----------|---------------|
| `basic_lending` | Easy | Minimal reasoning. Income pre-revealed, clear signals, 6-month Phase 2. |
| `noisy_signals` | Medium | Strategic information gathering. Conflicting features, extra penalty for wrong doc requests. |
| `adversarial_portfolio` | Hard | Long-horizon intervention timing. Borderline approval, elevated risk, capped signal quality (0.75 max), 3-miss default threshold. |

---

### 🔹 Reward Function

Includes immediate step costs and terminal outcome rewards.

| Event / Action          | Reward |
| ----------------------- | ------ |
| Phase 1: Correct Approve | +1.0   |
| Phase 1: Wrong Approve   | -2.0   |
| Phase 1: Correct Reject  | +0.6   |
| Phase 1: Wrong Reject    | -1.0   |
| Phase 1: Doc Request     | -0.10  |
| Phase 1: Flag Review     | -0.15  |
| Phase 2: Loan Repaid     | +1.5   |
| Phase 2: Loan Default    | -2.5   |
| Phase 2: Send Reminder   | -0.05  |
| Phase 2: Restructure     | -0.20  |
| Phase 2: Escalate        | -0.50  |

---

## 🏗️ Architecture

* **Server**: FastAPI-based OpenEnv environment
* **Client**: EnvClient wrapper (`client.py`) that handles dynamic two-phase routing of observations transparently.
* **Environment Logic**: `microfinance_env_environment.py` with Phase 1 and Phase 2 loops.
* **Dataset**: Synthetic microfinance profile generator mimicking real-world emerging market dynamics.
* **Grader**: A programmatic and LLM trajectory-aware grader (`grader.py`).

---

## 📦 Project Structure

```text
microfinance_env/
│
├── inference.py              # Baseline LLM inference (runs all 3 tasks)
├── models.py                 # Action, Phase, Observation, State definitions
├── client.py                 # Client wrapper handling Two-Phase returns
├── openenv.yaml              # OpenEnv spec (tasks, metadata)
│
├── server/
│   ├── app.py                # FastAPI server (OpenEnv interface)
│   ├── data_generator.py     # Synthetic dataset generator
│   ├── grader.py             # Trajectory-aware Grader evaluation
│   ├── reward_engine.py      # Isolated reward computation module
│   ├── microfinance_env_environment.py # Core environment (Phase 1 & Phase 2)
│   └── Dockerfile            # Container deployment
│
├── test_env.py               # Standalone test suite
├── requirements.txt
├── README.md
├── ABOUT.md
```

## 🧪 Running Locally

### 1. Activate environment
```bash
venv\Scripts\activate
```

### 2. Start server
```bash
uvicorn microfinance_env.server.app:app --reload
```
Server runs at: `http://localhost:8000`

---

## 🔌 API Endpoints

* `POST /reset` → start new episode
* `POST /step` → take action
* `GET /state` → inspect internal state
* `GET /docs` → Swagger UI
* `WS /ws` → WebSocket (low latency)

---

## 💻 Example Usage

```python
import asyncio
from client import MicrofinanceEnv
from models import CreditAction

async def main():
    async with MicrofinanceEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        
        # Phase 1 — application
        result = await env.step(CreditAction(action_type="REQUEST_INCOME_PROOF"))
        result = await env.step(CreditAction(action_type="REQUEST_CREDIT_HISTORY"))
        result = await env.step(CreditAction(action_type="APPROVE", rationale="Looks good"))
        
        # Phase 2 — monitoring begins
        for month in range(12):
            if result.done:
                break
            
            # Simple intervention policy
            action = "SEND_REMINDER" if result.observation.cumulative_misses > 0 else "DO_NOTHING"
            result = await env.step(CreditAction(action_type=action))
            print(f"Outcome: {result.observation.last_action_result}, Reward: {result.reward}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🐳 Docker

```bash
docker build -t microfinance-env -f server/Dockerfile .
docker run -p 8000:8000 microfinance-env
```

---

## ☁️ Deploy (OpenEnv)

```bash
openenv push
```

---

## 🎯 Why This Matters

This environment moves beyond static classification to highlight:
* **Two-Phase Lifecycle Planning**: Short-term Phase 1 decisions carry long-term Phase 2 consequences.
* **Information Quality vs Cost**: Paying for information directly improves future observational reliability.
* **Time-Sensitive Interventions**: The timing of an action matters as much as the action itself.

---

## 📌 Status
✅ Version 2.0 — Two-Phase Implementation complete. Ready for submission.

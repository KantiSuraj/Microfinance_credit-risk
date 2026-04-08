"""
inference.py — Baseline inference script for the Microfinance Credit Decision Environment.
===========================================================================================
MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Runs all 3 tasks (basic_lending, noisy_signals, adversarial_portfolio) sequentially
against a running local server instance.
"""

import os
import sys
import json
import time
import subprocess
import textwrap
import httpx
from typing import List, Optional

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:8000")
IMAGE_NAME   = os.getenv("IMAGE_NAME", "microfinance-env")

BENCHMARK    = "microfinance-env"


TASKS = [
    {
        "name": "basic_lending",
        "max_phase1_steps": 3,    # easy: decide quickly
        "max_phase2_steps": 6,
        "description": "Easy: clear signals, pre-revealed income, short Phase 2.",
    },
    {
        "name": "noisy_signals",
        "max_phase1_steps": 5,    # medium: need strategic doc requests
        "max_phase2_steps": 12,
        "description": "Medium: conflicting signals, must gather right documents.",
    },
    {
        "name": "adversarial_portfolio",
        "max_phase1_steps": 5,    # hard: borderline case
        "max_phase2_steps": 12,
        "description": "Hard: borderline approval, Phase 2 intervention timing critical.",
    },
]

TEMPERATURE = 0.7
MAX_TOKENS  = 300

SUCCESS_THRESHOLD = 0.40  # normalised score in [0, 1]


# ── Docker Control ────────────────────────────────────────────────────────

def start_container():
    print("[INFO] Starting Docker container...", file=sys.stderr, flush=True)

    container_id = subprocess.check_output([
        "docker", "run", "-d", "-p", "8000:8000", IMAGE_NAME
    ]).decode().strip()

    return container_id


def wait_for_server():
    print("[INFO] Waiting for server...", file=sys.stderr, flush=True)

    for _ in range(30):
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=2.0)
            if r.status_code == 200:
                print("[INFO] Server is ready!", file=sys.stderr, flush=True)
                return
        except:
            pass
        time.sleep(1)

    raise RuntimeError("Server did not start")


def stop_container(container_id):
    print("[INFO] Stopping container...", file=sys.stderr, flush=True)
    subprocess.run(["docker", "stop", container_id], stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", container_id], stdout=subprocess.DEVNULL)


# ── Stdout logging (exact format per spec) ────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM interaction ──────────────────────────────────────────────────────

SYSTEM_PROMPT_PHASE1 = textwrap.dedent("""\
You are a microfinance loan officer AI. You are evaluating a loan application.

Your available actions (respond with EXACTLY one JSON object):
- {"action": "REQUEST_INCOME_PROOF", "reasoning": "<why>"}
- {"action": "REQUEST_CREDIT_HISTORY", "reasoning": "<why>"}
- {"action": "FLAG_FOR_REVIEW", "reasoning": "<why>"}
- {"action": "APPROVE", "reasoning": "<why>"}
- {"action": "REJECT", "reasoning": "<why>"}

Strategy:
1. If key information is missing (income, credit history), request it first.
2. Weigh loan-to-income ratio, past defaults, repayment streak, occupation stability.
3. If borderline, flag for senior review before deciding.
4. Every document request costs money — don't over-request.
5. Respond with ONLY the JSON object. No extra text.
""").strip()

SYSTEM_PROMPT_PHASE2 = textwrap.dedent("""\
You are monitoring an approved microfinance loan. Each month you observe a payment status.

Your available actions (respond with EXACTLY one JSON object):
- {"action": "DO_NOTHING", "reasoning": "<why>"}
- {"action": "SEND_REMINDER", "reasoning": "<why>"}
- {"action": "RESTRUCTURE_LOAN", "reasoning": "<why>"}
- {"action": "ESCALATE_TO_RECOVERY", "reasoning": "<why>"}

Strategy:
1. If payments are on-time, DO_NOTHING to avoid unnecessary intervention costs.
2. After 1 missed payment, SEND_REMINDER (cheap, small risk reduction).
3. After 2+ consecutive misses or high cumulative misses, RESTRUCTURE_LOAN (costly but effective).
4. Only ESCALATE_TO_RECOVERY if the loan is clearly unrecoverable.
5. Consider signal_quality — lower quality means observations may be wrong.
6. Respond with ONLY the JSON object. No extra text.
""").strip()


def build_phase1_prompt(obs: dict, step: int) -> str:
    return textwrap.dedent(f"""\
    Step {step} — Phase 1 (Application)

    Applicant ID: {obs.get('applicant_id', 'N/A')}
    Occupation: {obs.get('occupation', 'N/A')}
    Region: {obs.get('region_tier', 'N/A')}
    Dependents: {obs.get('dependents', 'N/A')}
    Loan Amount: ₹{obs.get('loan_amount_requested', 0):,.0f}
    Monthly Income: {'₹{:,.0f}'.format(obs.get('monthly_income')) if obs.get('monthly_income') is not None else 'HIDDEN — request income proof'}
    Income Stability: {obs.get('income_source_stability') or 'HIDDEN'}
    Previous Loans: {obs.get('previous_loans') if obs.get('previous_loans') is not None else 'HIDDEN — request credit history'}
    Past Defaults: {obs.get('past_defaults') if obs.get('past_defaults') is not None else 'HIDDEN'}
    Repayment Streak: {obs.get('repayment_streak') if obs.get('repayment_streak') is not None else 'HIDDEN'}
    Senior Review: {obs.get('senior_review_comment') or 'Not requested'}
    Documents Submitted: {obs.get('documents_submitted', [])}
    Info Confidence: {obs.get('info_confidence', 0):.0%}
    Steps Used: {obs.get('step_count', 0)}/{obs.get('max_steps', 7)}

    Last result: {obs.get('last_action_result', 'N/A')}

    Decide your next action. Respond with exactly one JSON object.
    """).strip()


def build_phase2_prompt(obs: dict, step: int) -> str:
    return textwrap.dedent(f"""\
    Step {step} — Phase 2 (Monitoring)

    Month: {obs.get('month_number', 0)}/{obs.get('month_number', 0) + obs.get('months_remaining', 0)}
    Observed Payment: {obs.get('observed_payment', 'N/A')}
    Signal Quality: {obs.get('signal_quality', 0):.0%} (higher = more reliable observations)
    Cumulative Misses: {obs.get('cumulative_misses', 0)}
    Missed Streak: {obs.get('missed_streak', 0)}
    On-Time Streak: {obs.get('ontime_streak', 0)}
    Payment History: {obs.get('payment_history', [])}
    Interventions So Far: {obs.get('intervention_history', [])}
    Months Remaining: {obs.get('months_remaining', 0)}

    Last result: {obs.get('last_action_result', 'N/A')}

    Decide your next action. Respond with exactly one JSON object.
    """).strip()


def get_llm_action(client: OpenAI, obs: dict, step: int, phase: str) -> dict:
    """Ask the LLM for a structured action decision."""
    if phase == "APPLICATION" or phase == "TERMINAL":
        system = SYSTEM_PROMPT_PHASE1
        user = build_phase1_prompt(obs, step)
    else:
        system = SYSTEM_PROMPT_PHASE2
        user = build_phase2_prompt(obs, step)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse JSON from response (handle markdown code blocks)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        return {
            "action_type": parsed.get("action", "DO_NOTHING"),
            "rationale": parsed.get("reasoning", ""),
        }
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", file=sys.stderr, flush=True)
        # Fallback: phase-appropriate default
        if phase == "APPLICATION":
            return {"action_type": "APPROVE", "rationale": "fallback"}
        return {"action_type": "DO_NOTHING", "rationale": "fallback"}


# ── Environment HTTP interaction ──────────────────────────────────────────

def env_reset(http_client: httpx.Client) -> dict:
    """POST /reset and return the observation dict."""
    resp = http_client.post(f"{SERVER_URL}/reset")
    resp.raise_for_status()
    return resp.json()


def env_step(http_client: httpx.Client, action_type: str, rationale: str) -> dict:
    """POST /step with action and return the full result dict."""
    resp = http_client.post(
        f"{SERVER_URL}/step",
        json={"action": {"action_type": action_type, "rationale": rationale}},
    )
    resp.raise_for_status()
    return resp.json()


# ── Main episode runner ───────────────────────────────────────────────────

def run_task(task_config: dict, llm_client: OpenAI, http_client: httpx.Client) -> float:
    """Run a single task episode and return the normalised score [0, 1]."""
    task_name = task_config["name"]
    max_p1    = task_config["max_phase1_steps"]
    max_p2    = task_config["max_phase2_steps"]
    max_steps = max_p1 + max_p2

    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(http_client)
        obs = result.get("observation", result)
        done = result.get("done", False)
        phase = obs.get("current_phase", "APPLICATION")

        for step in range(1, max_steps + 1):
            if done:
                break

            action_dict = get_llm_action(llm_client, obs, step, phase)
            action_type = action_dict["action_type"]
            rationale   = action_dict["rationale"]

            result = env_step(http_client, action_type, rationale)
            obs    = result.get("observation", {})
            reward = result.get("reward", 0.0)
            done   = result.get("done", False)
            error  = obs.get("last_action_error", None)
            
            # Switch to Phase 2 promptly if the transition flag is raised
            if obs.get("transitioning_to_phase2", False):
                phase = "MONITORING"
            else:
                phase = obs.get("current_phase", phase)

            rewards.append(reward)
            steps_taken = step

            # Build action string for logging (structured JSON)
            action_str = json.dumps({"action": action_type, "reasoning": rationale}, ensure_ascii=False)

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Compute normalised score [0, 1] from terminal reward
        # Terminal reward range is approx [-3.0, 2.5], normalise to [0, 1]
        if rewards:
            terminal_reward = rewards[-1]
            score = (terminal_reward + 3.0) / 5.5  # maps [-3.0, 2.5] → [0.0, 1.0]
            score = min(max(score, 0.0), 1.0)       # clamp
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} failed: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    """Run all 3 tasks sequentially."""
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY environment variable not set.", file=sys.stderr, flush=True)
        sys.exit(1)

    llm_client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    http_client = httpx.Client(timeout=30.0)

    print(f"[INFO] Running inference against {SERVER_URL}", file=sys.stderr, flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[INFO] Tasks: {[t['name'] for t in TASKS]}", file=sys.stderr, flush=True)
    print("", file=sys.stderr, flush=True)

    scores = {}
    for task in TASKS:
        scores[task["name"]] = run_task(task, llm_client, http_client)
        print("", file=sys.stderr, flush=True)

    # Summary
    print("=" * 60, file=sys.stderr, flush=True)
    print("INFERENCE SUMMARY", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)
    for name, sc in scores.items():
        print(f"  {name:30s}: {sc:.2f}", file=sys.stderr, flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':30s}: {avg:.2f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

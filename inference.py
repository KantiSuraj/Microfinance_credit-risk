"""
inference.py — Inference script for the Microfinance Credit Decision Environment.
==================================================================================
MANDATORY ENV VARS:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    The Docker image name for the environment server.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Flow:
    inference.py
        -> Docker (manual start/stop via subprocess)
        -> MicrofinanceEnv (HTTP EnvClient from SDK)
        -> OpenAI (HF API)
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from typing import List, Optional, Union

import httpx
from openai import OpenAI

from models import (
    CreditAction,                         
    ApplicantObservation,      
    MonitoringObservation,     
)
from client import (
    MicrofinanceEnv
)


# ── Configuration ──────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME   = os.getenv("IMAGE_NAME")         
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:8000")

BENCHMARK         = "microfinance-env"
TEMPERATURE       = 0.7
MAX_TOKENS        = 300
SUCCESS_THRESHOLD = 0.40     # normalised score in [0, 1]

TASKS = [
    {
        "name": "basic_lending",
        "max_phase1_steps": 3,
        "max_phase2_steps": 6,
        "description": "Easy: clear signals, pre-revealed income, short Phase 2.",
    },
    {
        "name": "noisy_signals",
        "max_phase1_steps": 5,
        "max_phase2_steps": 12,
        "description": "Medium: conflicting signals, must gather right documents.",
    },
    {
        "name": "adversarial_portfolio",
        "max_phase1_steps": 5,
        "max_phase2_steps": 12,
        "description": "Hard: borderline approval, Phase 2 intervention timing critical.",
    },
]


# ── Docker management ──────────────────────────────────────────────────────────
# MicrofinanceEnv extends EnvClient (HTTP-only). There is no from_docker_image().
# Docker must be managed manually here.

def start_container(image_name: str) -> str:
    """Start the Docker container and return its ID."""
    print("[INFO] Starting Docker container...", file=sys.stderr, flush=True)
    container_id = subprocess.check_output([
        "docker", "run", "-d", "-p", "8000:8000", image_name
    ]).decode().strip()
    print(f"[INFO] Container: {container_id[:12]}", file=sys.stderr, flush=True)
    return container_id


def wait_for_server(url: str, retries: int = 30) -> None:
    """Block until /health returns 200 or raise on timeout."""
    print("[INFO] Waiting for server...", file=sys.stderr, flush=True)
    for _ in range(retries):
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                print("[INFO] Server is ready.",file=sys.stderr, flush=True)
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not become ready after {retries}s")


def stop_container(container_id: str) -> None:
    """Stop and remove the Docker container."""
    print("[INFO] Stopping container...", file=sys.stderr, flush=True)
    subprocess.run(["docker", "stop", container_id], stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "rm",   container_id], stdout=subprocess.DEVNULL)


# ── Stdout logging (exact spec format) ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
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


# ── LLM prompts ────────────────────────────────────────────────────────────────

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
3. After 2+ consecutive misses or high cumulative misses, RESTRUCTURE_LOAN.
4. Only ESCALATE_TO_RECOVERY if the loan is clearly unrecoverable.
5. Consider signal_quality — lower quality means observations may be unreliable.
6. Respond with ONLY the JSON object. No extra text.
""").strip()


def build_phase1_prompt(obs: ApplicantObservation, step: int) -> str:
    """Build user prompt from the Pydantic ApplicantObservation object."""
    return textwrap.dedent(f"""\
    Step {step} — Phase 1 (Application)

    Applicant ID: {obs.applicant_id}
    Occupation: {obs.occupation or 'N/A'}
    Region: {obs.region_tier or 'N/A'}
    Dependents: {obs.dependents if obs.dependents is not None else 'N/A'}
    Loan Amount: {f'Rs.{obs.loan_amount_requested:,.0f}' if obs.loan_amount_requested is not None else 'N/A'}
    Monthly Income: {f'Rs.{obs.monthly_income:,.0f}' if obs.monthly_income is not None else 'HIDDEN — request income proof'}
    Income Stability: {obs.income_source_stability or 'HIDDEN'}
    Previous Loans: {obs.previous_loans if obs.previous_loans is not None else 'HIDDEN — request credit history'}
    Past Defaults: {obs.past_defaults if obs.past_defaults is not None else 'HIDDEN'}
    Repayment Streak: {obs.repayment_streak if obs.repayment_streak is not None else 'HIDDEN'}
    Senior Review: {obs.senior_review_comment or 'Not requested'}
    Documents Submitted: {obs.documents_submitted}
    Info Confidence: {obs.info_confidence:.0%}
    Steps Used: {obs.step_count}/{obs.max_steps}

    Last result: {obs.last_action_result or 'N/A'}

    Decide your next action. Respond with exactly one JSON object.
    """).strip()


def build_phase2_prompt(obs: MonitoringObservation, step: int) -> str:
    """Build user prompt from the Pydantic MonitoringObservation object."""
    total_months = obs.month_number + obs.months_remaining
    return textwrap.dedent(f"""\
    Step {step} — Phase 2 (Monitoring)

    Month: {obs.month_number}/{total_months}
    Observed Payment: {obs.observed_payment}
    Signal Quality: {obs.signal_quality:.0%}
    Cumulative Misses: {obs.cumulative_misses}
    Missed Streak: {obs.missed_streak}
    On-Time Streak: {obs.ontime_streak}
    Payment History: {obs.payment_history}
    Interventions So Far: {obs.intervention_history}
    Months Remaining: {obs.months_remaining}

    Last result: {obs.last_action_result or 'N/A'}

    Decide your next action. Respond with exactly one JSON object.
    """).strip()


# ── LLM interaction ────────────────────────────────────────────────────────────

def get_llm_action(
    client: OpenAI,
    obs: Union[ApplicantObservation, MonitoringObservation],
    step: int,
) -> dict:
    """
    Ask the LLM for a structured action decision.
    Routes to the correct prompt/system based on the obs type.
    Returns {"action_type": ..., "rationale": ...}.
    """
    if isinstance(obs, ApplicantObservation):
        system = SYSTEM_PROMPT_PHASE1
        user   = build_phase1_prompt(obs, step)
        fallback_action = "APPROVE"
    else:
        system = SYSTEM_PROMPT_PHASE2
        user   = build_phase2_prompt(obs, step)
        fallback_action = "DO_NOTHING"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if model wraps response in ```json ... ```
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        return {
            "action_type": parsed.get("action", fallback_action),
            "rationale":   parsed.get("reasoning", ""),
        }

    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}",  file=sys.stderr, flush=True)
        return {"action_type": fallback_action, "rationale": "fallback"}


# ── Task runner ────────────────────────────────────────────────────────────────

def run_task(task_config: dict, llm_client: OpenAI) -> float:
    """
    Run one full episode for a task.

    Uses MicrofinanceEnv as a synchronous context manager.
    obs is always a Pydantic model (ApplicantObservation or MonitoringObservation),
    so we access fields directly (obs.field_name), never obs.get().

    Returns normalised score in [0, 1].
    """
    task_name = task_config["name"]
    max_steps = task_config["max_phase1_steps"] + task_config["max_phase2_steps"]

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # MicrofinanceEnv is an EnvClient — use .sync() context manager
        with MicrofinanceEnv(base_url=SERVER_URL).sync() as env:
            result = env.reset()
            obs    = result.observation    # ApplicantObservation (Phase 1 start)
            done   = result.done

            for step in range(1, max_steps + 1):
                if done:
                    break

                action_dict = get_llm_action(llm_client, obs, step)
                action_type = action_dict["action_type"]
                rationale   = action_dict["rationale"]

                # CreditAction field names confirmed from models.py:
                #   action_type: ActionType  (the action string)
                #   rationale:   str         (the reasoning)
                result = env.step(CreditAction(
                    action_type=action_type,
                    rationale=rationale,
                ))

                obs    = result.observation   # may switch to MonitoringObservation in Phase 2
                reward = result.reward
                done   = result.done

                # last_action_result is a field on the Pydantic obs, not an error key
                # We only surface it when it looks like an error (non-empty, non-success text)
                last_result = getattr(obs, "last_action_result", "") or ""
                error = last_result if last_result else None

                rewards.append(reward)
                steps_taken = step

                action_str = json.dumps(
                    {"action": action_type, "reasoning": rationale},
                    ensure_ascii=False,
                )
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

        # Normalise terminal reward from approx [-3.0, 2.5] -> [0.0, 1.0]
        if rewards:
            score = (rewards[-1] + 3.0) / 5.5
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    if not IMAGE_NAME:
        print("[ERROR] IMAGE_NAME environment variable not set.", flush=True)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Model  : {MODEL_NAME}",file=sys.stderr, flush=True)
    print(f"[INFO] Image  : {IMAGE_NAME}",file=sys.stderr, flush=True)
    print(f"[INFO] Server : {SERVER_URL}", file=sys.stderr,flush=True)
    print(f"[INFO] Tasks  : {[t['name'] for t in TASKS]}",file=sys.stderr, flush=True)
    print("",file=sys.stderr, flush=True)

    container_id = start_container(IMAGE_NAME)

    try:
        wait_for_server(SERVER_URL)

        scores = {}
        for task in TASKS:
            scores[task["name"]] = run_task(task, llm_client)
            print("",file=sys.stderr, flush=True)

    finally:
        # Always stop the container, even if a task crashes
        stop_container(container_id)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("INFERENCE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for name, sc in scores.items():
        print(f"  {name:30s}: {sc:.2f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':30s}: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()
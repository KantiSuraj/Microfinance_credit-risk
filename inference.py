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
#
# Change 1: _find_free_port and _kill_process_on_port removed entirely.
# They introduced subprocess calls to OS binaries (netstat, ss, fuser, lsof,
# taskkill) that are not present in the minimal Linux evaluator container.
# The evaluator manages the Docker container externally; port management
# is not our responsibility and must not crash the script.


def start_container(image_name: str) -> str:
    """Start the Docker container and return its ID.

    Change 2: All hard raises replaced with warn-and-return-empty so the
    script degrades gracefully when Docker is unavailable or already managed
    externally by the evaluator.
    """
    print("[INFO] Starting Docker container...", file=sys.stderr, flush=True)

    # Guard: image name must be non-empty
    if not image_name or not image_name.strip():
        print("[WARN] IMAGE_NAME is empty or None; skipping docker run.",
              file=sys.stderr, flush=True)
        return ""

    # Attempt docker run — non-fatal on any failure
    try:
        container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", "8000:8000", image_name],
            stderr=subprocess.PIPE,
        ).decode().strip()
    except subprocess.CalledProcessError as exc:
        stderr_msg = exc.stderr.decode().strip() if exc.stderr else ""
        print(
            f"[WARN] docker run failed (exit {exc.returncode}): {stderr_msg}",
            file=sys.stderr, flush=True,
        )
        return ""
    except FileNotFoundError:
        print(
            "[WARN] Docker executable not found — assuming server is already running.",
            file=sys.stderr, flush=True,
        )
        return ""
    except Exception as exc:
        print(f"[WARN] docker run raised unexpected error: {exc}",
              file=sys.stderr, flush=True)
        return ""

    if not container_id:
        print("[WARN] docker run returned empty container ID.",
              file=sys.stderr, flush=True)
        return ""

    print(f"[INFO] Container: {container_id[:12]}", file=sys.stderr, flush=True)
    return container_id


def wait_for_server(url: str, timeout_seconds: int = 90) -> bool:
    """Block until /health returns 200 or timeout.

    Returns True if server became ready, False otherwise.
    Never raises — all exceptions are caught and logged.
    """
    print(f"[INFO] Waiting for server (timeout={timeout_seconds}s)...",
          file=sys.stderr, flush=True)
    last_exc: Optional[Exception] = None
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            r = httpx.get(f"{url}/health", timeout=3.0)
            if r.status_code == 200:
                print(f"[INFO] Server is ready (attempt {attempt}).",
                      file=sys.stderr, flush=True)
                return True
            # Non-200 but server responded — keep waiting
        except (httpx.TransportError, httpx.TimeoutException,
                ConnectionError, OSError) as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        time.sleep(2)
    print(
        f"[ERROR] Server at {url} did not become ready after {timeout_seconds}s. "
        f"Last error: {last_exc}",
        file=sys.stderr, flush=True,
    )
    return False


def stop_container(container_id: str) -> None:
    """Stop and remove the Docker container."""
    if not container_id:
        print("[WARN] stop_container called with empty container_id; skipping.",
              file=sys.stderr, flush=True)
        return
    print("[INFO] Stopping container...", file=sys.stderr, flush=True)
    try:
        subprocess.run(
            ["docker", "stop", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("[WARN] 'docker stop' timed out; forcing kill.", file=sys.stderr, flush=True)
        subprocess.run(
            ["docker", "kill", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"[WARN] 'docker stop' raised: {exc}", file=sys.stderr, flush=True)

    try:
        subprocess.run(
            ["docker", "rm", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except Exception as exc:
        print(f"[WARN] 'docker rm' raised: {exc}", file=sys.stderr, flush=True)


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
6. IMPORTANT: Using the SAME action for 4+ consecutive months incurs escalating
   penalties. Vary your strategy based on changing conditions — e.g. switch from
   DO_NOTHING to SEND_REMINDER periodically even during on-time streaks.
7. Watch for economic_stress_signal > 0.3 — it warns of a possible shock next month.
8. Respond with ONLY the JSON object. No extra text.
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
    force_phase2: bool = False,
) -> dict:
    """
    Ask the LLM for a structured action decision.
    Routes to the correct prompt/system based on the obs type.
    force_phase2 overrides isinstance routing for the transition step.
    Returns {"action_type": ..., "rationale": ...}.
    """
    if force_phase2 or isinstance(obs, MonitoringObservation):
        system = SYSTEM_PROMPT_PHASE2
        if isinstance(obs, MonitoringObservation):
            user = build_phase2_prompt(obs, step)
        else:
            # Transition step: obs is still ApplicantObservation but we're
            # conceptually in Phase 2.  Provide a minimal Phase 2 prompt.
            user = (
                f"Step {step} — Phase 2 (Monitoring)\n\n"
                f"Loan has been approved and monitoring begins.\n"
                f"Decide your first monitoring action. "
                f"Respond with exactly one JSON object."
            )
        fallback_action = "DO_NOTHING"
    else:
        system = SYSTEM_PROMPT_PHASE1
        user   = build_phase1_prompt(obs, step)
        fallback_action = "APPROVE"

    try:
        # Change 5: bound the OpenAI call with a timeout to prevent watchdog kill
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            timeout=20,
        )

        # Guard: completion object, choices list, or message may be None/empty
        if (
            not completion
            or not getattr(completion, "choices", None)
            or completion.choices[0] is None
            or getattr(completion.choices[0], "message", None) is None
        ):
            print("[WARN] LLM returned empty/malformed completion; using fallback.",
                  file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-empty-completion"}

        text = (completion.choices[0].message.content or "").strip()

        # Guard: empty content string
        if not text:
            print("[WARN] LLM returned empty content string; using fallback.",
                  file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-empty-content"}

        # Strip markdown code fences if model wraps response in ```json ... ```
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else ""
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        # Guard: text empty after fence-stripping
        if not text:
            print("[WARN] LLM text empty after fence-strip; using fallback.",
                  file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-empty-after-strip"}

        # Guard: JSON parse may fail
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as jexc:
            print(f"[WARN] JSON parse failed ({jexc}); raw={text!r}; using fallback.",
                  file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-json-error"}

        # Guard: parsed must be a dict to safely call .get()
        if not isinstance(parsed, dict):
            print(f"[WARN] LLM JSON not a dict ({type(parsed).__name__}); using fallback.",
                  file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-not-dict"}

        action_type = parsed.get("action", fallback_action)
        rationale   = parsed.get("reasoning", "")

        # Guard: action_type must be a non-empty string
        if not isinstance(action_type, str) or not action_type.strip():
            action_type = fallback_action
        # Guard: rationale must be a string
        if not isinstance(rationale, str):
            rationale = str(rationale)

        return {
            "action_type": action_type,
            "rationale":   rationale,
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

    # ── Local episode tracking ─────────────────────────────────────────────
    in_phase2:              bool        = False
    phase1_decision:        str         = "REJECT"
    reached_phase2:         bool        = False
    docs_collected_local:   List[str]   = []
    phase1_steps_local:     int         = 0
    phase1_reward_local:    Optional[float] = None
    signal_quality_local:   Optional[float] = None
    payment_history_local:  List[str]   = []
    intervention_local:     List[str]   = []
    terminal_outcome:       str         = "TIMEOUT"
    default_prob_m3:        Optional[float] = None
    default_prob_m6:        Optional[float] = None
    phase2_month_counter:   int         = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Wrap the connection itself so a dead server doesn't crash inference.
        try:
            env_ctx = MicrofinanceEnv(base_url=SERVER_URL).sync()
            env = env_ctx.__enter__()
        except Exception as conn_exc:
            print(
                f"[ERROR] Cannot connect to environment server: {conn_exc}",
                file=sys.stderr, flush=True,
            )
            # Change 6: no re-raise — return safely with zero score
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return 0.0

        try:
            # ── Switch the server to the correct task difficulty ────────────
            try:
                switch_resp = httpx.post(
                    f"{SERVER_URL}/set_task",
                    json={"task_name": task_name},
                    timeout=10.0,
                )
                if switch_resp.status_code != 200:
                    print(
                        f"[WARN] /set_task returned {switch_resp.status_code}: {switch_resp.text}",
                        file=sys.stderr, flush=True,
                    )
                else:
                    print(f"[INFO] Task set to '{task_name}'.", file=sys.stderr, flush=True)
            except (httpx.TransportError, httpx.TimeoutException,
                    ConnectionError, OSError) as exc:
                print(f"[WARN] /set_task connection failed: {exc}", file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"[WARN] /set_task request failed: {exc}", file=sys.stderr, flush=True)

            # Guard: env.reset() may raise or return None
            try:
                result = env.reset()
            except (httpx.TransportError, httpx.TimeoutException,
                    ConnectionError, OSError) as exc:
                print(f"[ERROR] env.reset() connection failed: {exc}",
                      file=sys.stderr, flush=True)
                # Change 6: no re-raise — return safely
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return 0.0
            except Exception as exc:
                print(f"[WARN] env.reset() failed: {exc}", file=sys.stderr, flush=True)
                # Change 6: suppress — return safely instead of re-raising
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return 0.0

            if result is None:
                print("[WARN] env.reset() returned None; cannot proceed.",
                      file=sys.stderr, flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return 0.0

            obs  = result.observation
            done = result.done

            if obs is None:
                print("[WARN] env.reset() returned result with None observation.",
                      file=sys.stderr, flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return 0.0

            for step in range(1, max_steps + 1):
                if done:
                    break

                action_dict = get_llm_action(
                    llm_client, obs, step, force_phase2=in_phase2,
                )
                action_type = action_dict.get("action_type", "APPROVE")
                rationale   = action_dict.get("rationale", "")

                # Guard: ensure action_type and rationale are plain strings
                if not isinstance(action_type, str) or not action_type.strip():
                    action_type = "DO_NOTHING" if in_phase2 else "APPROVE"
                if not isinstance(rationale, str):
                    rationale = str(rationale)

                # Prevent Phase 1 actions leaking into Phase 2
                if in_phase2 and action_type in (
                    "APPROVE", "REJECT", "REQUEST_INCOME_PROOF",
                    "REQUEST_CREDIT_HISTORY", "FLAG_FOR_REVIEW",
                ):
                    print(
                        f"[WARN] Phase 1 action '{action_type}' in Phase 2; "
                        f"overriding to DO_NOTHING.",
                        file=sys.stderr, flush=True,
                    )
                    action_type = "DO_NOTHING"
                    rationale = "Auto-corrected: Phase 1 action in Phase 2"

                try:
                    result = env.step(CreditAction(
                        action_type=action_type,
                        rationale=rationale,
                    ))
                except (httpx.TransportError, httpx.TimeoutException,
                        ConnectionError, OSError) as exc:
                    print(
                        f"[ERROR] env.step() connection failed at step {step}: {exc}",
                        file=sys.stderr, flush=True,
                    )
                    # Treat connection loss as episode end — score what we have
                    break
                except Exception as exc:
                    print(f"[WARN] env.step() failed at step {step}: {exc}",
                          file=sys.stderr, flush=True)
                    # Change 6: suppress instead of re-raise
                    break

                if result is None:
                    print(f"[WARN] env.step() returned None at step {step}; ending episode.",
                          file=sys.stderr, flush=True)
                    break

                obs    = result.observation
                reward = result.reward
                done   = result.done

                # Guard: reward must be a finite float
                try:
                    reward = float(reward)
                    if not (reward == reward):   # NaN check
                        reward = 0.0
                except (TypeError, ValueError):
                    reward = 0.0

                # Guard: done must be a bool
                if not isinstance(done, bool):
                    done = bool(done)

                last_result = ""
                if obs is not None:
                    try:
                        last_result = getattr(obs, "last_action_result", "") or ""
                    except Exception:
                        last_result = ""
                error = last_result if last_result else None

                rewards.append(reward)
                steps_taken = step

                # ── Local episode state tracking ──────────────────────────
                if not in_phase2:
                    phase1_steps_local = step

                    if action_type == "REQUEST_INCOME_PROOF" and "income_proof" not in docs_collected_local:
                        docs_collected_local.append("income_proof")
                    elif action_type == "REQUEST_CREDIT_HISTORY" and "credit_history" not in docs_collected_local:
                        docs_collected_local.append("credit_history")
                    elif action_type == "FLAG_FOR_REVIEW" and "senior_review" not in docs_collected_local:
                        docs_collected_local.append("senior_review")

                    if action_type == "APPROVE":
                        phase1_decision = "APPROVE"
                        phase1_reward_local = reward
                        reached_phase2 = True
                        transitioning = getattr(obs, "transitioning_to_phase2", False)
                        if transitioning:
                            in_phase2 = True
                        print(
                            f"[DEBUG] APPROVE response: "
                            f"transitioning={transitioning} "
                            f"last_result={last_result[:100]}",
                            file=sys.stderr, flush=True,
                        )

                    elif action_type == "REJECT":
                        phase1_decision = "REJECT"
                        terminal_outcome = "REJECTED"

                else:
                    if isinstance(obs, MonitoringObservation):
                        payment_history_local = list(getattr(obs, "payment_history", []))
                        intervention_local = list(getattr(obs, "intervention_history", []))

                        if signal_quality_local is None:
                            sq_field = getattr(obs, "signal_quality", None)
                            if sq_field is not None:
                                signal_quality_local = float(sq_field)
                            else:
                                print(
                                    "[WARN] signal_quality not found on MonitoringObservation",
                                    file=sys.stderr, flush=True,
                                )

                        phase2_month_counter += 1
                        hidden_prob = getattr(obs, "current_default_prob_hidden", None)
                        if hidden_prob is not None:
                            if phase2_month_counter == 3:
                                default_prob_m3 = hidden_prob
                            elif phase2_month_counter == 6:
                                default_prob_m6 = hidden_prob

                    if done and last_result:
                        msg_lower = last_result.lower()
                        if "fully repaid" in msg_lower or "repaid" in msg_lower:
                            terminal_outcome = "REPAID"
                        elif "default" in msg_lower:
                            terminal_outcome = "DEFAULT"
                        elif "escalat" in msg_lower:
                            terminal_outcome = "ESCALATED"

                # Fallback phase transition detection from obs type
                if isinstance(obs, MonitoringObservation) and not in_phase2:
                    in_phase2 = True
                    if not reached_phase2:
                        reached_phase2 = True
                        phase1_decision = "APPROVE"

                # Serialise action to JSON string for the log line (spec requires JSON)
                try:
                    action_log_str = json.dumps(
                        {"action": action_type, "reasoning": rationale},
                        ensure_ascii=False,
                    )
                except (TypeError, ValueError):
                    action_log_str = json.dumps(
                        {"action": str(action_type), "reasoning": str(rationale)},
                        ensure_ascii=False,
                    )
                log_step(step=step, action=action_log_str, reward=reward, done=done, error=error)

                if done:
                    break

                if obs is None:
                    print(f"[WARN] obs is None after step {step}; ending episode.",
                          file=sys.stderr, flush=True)
                    break

        finally:
            # Always close the env context manager
            try:
                env_ctx.__exit__(None, None, None)
            except Exception:
                pass

        # ── Score via grader (authoritative) ──────────────────────────────────
        try:
            state_resp = httpx.get(f"{SERVER_URL}/state", timeout=10.0)
            if state_resp.status_code == 200:
                # Guard: response body may not be valid JSON
                try:
                    state_data = state_resp.json()
                except Exception as jexc:
                    print(f"[WARN] /state body not valid JSON ({jexc}); using reward fallback.",
                          file=sys.stderr, flush=True)
                    if rewards:
                        score = (rewards[-1] + 3.0) / 5.5
                        score = min(max(score, 0.0), 1.0)
                    state_data = None

                # Guard: state_data must be a dict
                if state_data is not None and not isinstance(state_data, dict):
                    print(f"[WARN] /state JSON is not a dict ({type(state_data).__name__}); using reward fallback.",
                          file=sys.stderr, flush=True)
                    if rewards:
                        score = (rewards[-1] + 3.0) / 5.5
                        score = min(max(score, 0.0), 1.0)
                    state_data = None

                if state_data is not None:
                    ground_truth = state_data.get("ground_truth_label", "REJECT")
                    default_prob = state_data.get("true_default_probability", 0.5)

                    terminal_reward = state_data.get("terminal_reward")
                    if terminal_reward is None and rewards:
                        terminal_reward = rewards[-1]
                    elif terminal_reward is None:
                        terminal_reward = 0.0

                    p1_reward = phase1_reward_local
                    if p1_reward is None:
                        p1_reward = state_data.get("phase1_reward", -1.5)

                    sq = signal_quality_local
                    if sq is None:
                        sq = state_data.get("signal_quality")

                    is_borderline = abs(default_prob - 0.50) < 0.20
                    has_conflicting = task_name == "noisy_signals"

                    episode_log = {
                        "phase1_decision":         phase1_decision,
                        "ground_truth":            ground_truth,
                        "default_prob":            default_prob,
                        "phase1_steps":            phase1_steps_local,
                        "docs_collected":          docs_collected_local,
                        "phase1_reward":           p1_reward,
                        "reached_phase2":          reached_phase2,
                        "terminal_outcome":        terminal_outcome,
                        "terminal_reward":         terminal_reward,
                        "payment_history":         payment_history_local,
                        "intervention_history":    intervention_local,
                        "signal_quality":          sq,
                        "is_borderline":           is_borderline,
                        "has_conflicting_signal":  has_conflicting,
                        "default_prob_at_month3":  default_prob_m3,
                        "default_prob_at_month6":  default_prob_m6,
                    }

                    print(
                        f"[DEBUG] episode_log: {json.dumps(episode_log, default=str)}",
                        file=sys.stderr, flush=True,
                    )

                    # Change 3: grader import is optional — not guaranteed in evaluator PYTHONPATH
                    grade = None
                    try:
                        from server.grader import grade_trajectory
                        grade = grade_trajectory(episode_log)
                    except Exception as grader_exc:
                        print(f"[WARN] Grader unavailable: {grader_exc}",
                              file=sys.stderr, flush=True)

                    if grade is not None:
                        # Guard: grade.score may be None, NaN, or non-numeric
                        try:
                            score = float(grade.score)
                            if score != score:   # NaN check
                                raise ValueError("grade.score is NaN")
                        except (TypeError, ValueError, AttributeError) as gexc:
                            print(f"[WARN] grade.score invalid ({gexc}); using reward fallback.",
                                  file=sys.stderr, flush=True)
                            if rewards:
                                score = (rewards[-1] + 3.0) / 5.5
                                score = min(max(score, 0.0), 1.0)

                        # Guard: sub-score attributes may be missing or non-numeric
                        try:
                            p1   = float(getattr(grade, "phase1_score",    0.0) or 0.0)
                            p2   = float(getattr(grade, "phase2_score",    0.0) or 0.0)
                            tim  = float(getattr(grade, "timing_score",    0.0) or 0.0)
                            info = float(getattr(grade, "info_flow_score", 0.0) or 0.0)
                            suff = float(getattr(grade, "info_sufficiency", 0.0) or 0.0)
                            print(
                                f"[INFO] Grader score={score:.4f} "
                                f"(p1={p1:.3f} p2={p2:.3f} "
                                f"timing={tim:.3f} info={info:.3f} "
                                f"suff={suff:.3f})",
                                file=sys.stderr, flush=True,
                            )
                        except Exception as pexc:
                            print(f"[INFO] Grader score={score:.4f} (sub-scores unavailable: {pexc})",
                                  file=sys.stderr, flush=True)
                    else:
                        # Grader not available — use reward normalisation
                        if rewards:
                            score = (rewards[-1] + 3.0) / 5.5
                            score = min(max(score, 0.0), 1.0)
            else:
                print(f"[WARN] /state returned {state_resp.status_code}; using reward fallback.",
                      file=sys.stderr, flush=True)
                if rewards:
                    score = (rewards[-1] + 3.0) / 5.5
                    score = min(max(score, 0.0), 1.0)
        except Exception as exc:
            print(f"[WARN] Grader scoring failed ({exc}); using reward fallback.",
                  file=sys.stderr, flush=True)
            if rewards:
                score = (rewards[-1] + 3.0) / 5.5
                score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        # Change 6: top-level catch — no unhandled exception escapes run_task
        print(f"[WARN] suppressed exception in run_task: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Change 4: do not exit on missing API key — let inference attempt proceed
    if not API_KEY:
        print("[ERROR] API key missing — inference may fail.", flush=True)

    # Change 4: do not exit on missing IMAGE_NAME — assume external server mode
    if not IMAGE_NAME:
        print("[WARN] IMAGE_NAME not set — assuming external server.", flush=True)

    # Guard: validate API_BASE_URL is a non-empty string
    if not API_BASE_URL or not API_BASE_URL.strip():
        print("[WARN] API_BASE_URL is empty or not set — inference may fail.", flush=True)

    # Guard: initialise OpenAI client safely
    try:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[ERROR] Failed to initialise OpenAI client: {exc}", flush=True)
        # Still attempt tasks — env interaction doesn't need the LLM client to init cleanly
        llm_client = None

    print(f"[INFO] Model  : {MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[INFO] Image  : {IMAGE_NAME}", file=sys.stderr, flush=True)
    print(f"[INFO] Server : {SERVER_URL}", file=sys.stderr, flush=True)
    print(f"[INFO] Tasks  : {[t['name'] for t in TASKS]}", file=sys.stderr, flush=True)
    print("", file=sys.stderr, flush=True)

    # Change 2: start_container is non-fatal — returns "" on any failure
    container_id = ""
    try:
        container_id = start_container(IMAGE_NAME) if IMAGE_NAME else ""
    except Exception as exc:
        print(f"[WARN] Docker unavailable or failed: {exc}", flush=True)
        container_id = ""

    try:
        # Change 7: wait_for_server failure is non-fatal
        server_ready = False
        try:
            server_ready = wait_for_server(SERVER_URL, timeout_seconds=90)
        except Exception as exc:
            print(f"[WARN] Server readiness check failed: {exc}", flush=True)

        if not server_ready:
            print("[WARN] Proceeding to tasks despite server not confirmed ready.",
                  file=sys.stderr, flush=True)

        scores = {}
        for task in TASKS:
            try:
                scores[task["name"]] = run_task(task, llm_client)
            except Exception as exc:
                # Change 6: individual task crash must not abort other tasks
                print(f"[WARN] suppressed exception in task '{task['name']}': {exc}",
                      file=sys.stderr, flush=True)
                scores[task["name"]] = 0.0
            print("", file=sys.stderr, flush=True)

    finally:
        # Always attempt container cleanup — non-fatal
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

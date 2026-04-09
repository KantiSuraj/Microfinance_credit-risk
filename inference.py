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

def _find_free_port() -> int:
    """Find a free TCP port on localhost to avoid bind conflicts.
    Pure-Python / cross-platform (Windows, macOS, Linux).
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _kill_process_on_port(port: int) -> None:
    """Attempt to free *port* using only cross-platform mechanisms.

    Strategy (no Unix-only tools like fuser/lsof):
      1. Pure-Python  — use psutil if available (works everywhere).
      2. Windows      — netstat + taskkill  (built-in since XP).
      3. Linux/macOS  — /proc/net/tcp6? + kill via os.kill  OR
                        ss/netstat fallback + os.kill.
    All paths are wrapped so a failure in one never crashes the caller.
    """
    import signal
    import platform

    pids_to_kill: List[int] = []

    # ── Path 1: psutil (cross-platform, preferred) ────────────────────────
    try:
        import psutil  # type: ignore
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                pids_to_kill.append(conn.pid)
    except ImportError:
        pass  # psutil not installed — fall through to OS-specific paths
    except Exception as exc:
        print(f"[WARN] psutil port scan failed: {exc}", file=sys.stderr, flush=True)

    # ── Path 2: Windows — netstat + taskkill ─────────────────────────────
    if not pids_to_kill and platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "TCP"],
                stderr=subprocess.DEVNULL,
            ).decode(errors="replace")
            for line in out.splitlines():
                parts = line.split()
                # e.g.  TCP  0.0.0.0:8000  0.0.0.0:0  LISTENING  1234
                if len(parts) >= 5 and f":{port}" in parts[1]:
                    try:
                        pids_to_kill.append(int(parts[-1]))
                    except ValueError:
                        pass
        except Exception as exc:
            print(f"[WARN] Windows netstat failed: {exc}", file=sys.stderr, flush=True)

    # ── Path 3: Linux/macOS — ss (iproute2) ──────────────────────────────
    if not pids_to_kill and platform.system() in ("Linux", "Darwin"):
        try:
            # ss -tlnp  (Linux iproute2)
            out = subprocess.check_output(
                ["ss", "-tlnp", f"sport = :{port}"],
                stderr=subprocess.DEVNULL,
            ).decode(errors="replace")
            # pid= appears as  users:(("python",pid=1234,fd=5))
            import re
            for match in re.finditer(r"pid=(\d+)", out):
                pids_to_kill.append(int(match.group(1)))
        except Exception:
            pass  # ss not available — try netstat -tlnp (macOS / older Linux)

    # ── Path 4: macOS / older Linux — netstat -tlnp ───────────────────────
    if not pids_to_kill and platform.system() in ("Linux", "Darwin"):
        try:
            flag = "-tlnp" if platform.system() == "Linux" else "-anv"
            out = subprocess.check_output(
                ["netstat", flag],
                stderr=subprocess.DEVNULL,
            ).decode(errors="replace")
            import re
            for line in out.splitlines():
                if f".{port} " in line or f":{port} " in line or f":{port}\t" in line:
                    # macOS netstat has PID in last column when -v is used
                    parts = line.split()
                    for part in reversed(parts):
                        try:
                            pids_to_kill.append(int(part))
                            break
                        except ValueError:
                            continue
        except Exception as exc:
            print(f"[WARN] netstat fallback failed: {exc}", file=sys.stderr, flush=True)

    # ── Kill collected PIDs ───────────────────────────────────────────────
    if not pids_to_kill:
        print(f"[WARN] Could not identify any PID holding port {port}.",
              file=sys.stderr, flush=True)
        return

    killed: List[int] = []
    for pid in set(pids_to_kill):   # deduplicate
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.3)
                # If still alive, force-kill
                try:
                    os.kill(pid, 0)          # check if process still exists
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass                     # already gone after SIGTERM
            killed.append(pid)
        except (PermissionError, ProcessLookupError) as exc:
            print(f"[WARN] Could not kill PID {pid}: {exc}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[WARN] Unexpected error killing PID {pid}: {exc}", file=sys.stderr, flush=True)

    if killed:
        print(f"[INFO] Freed port {port} by terminating PID(s): {killed}",
              file=sys.stderr, flush=True)
        time.sleep(1)   # brief pause so the OS releases the port socket


def start_container(image_name: str) -> str:
    """Start the Docker container and return its ID."""
    print("[INFO] Starting Docker container...", file=sys.stderr, flush=True)

    # ── Guard: port 8000 may already be in use ─────────────────────────────
    import socket
    port_in_use = False
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(1)
        probe.connect(("localhost", 8000))
        probe.close()
        port_in_use = True
    except Exception:
        port_in_use = False

    if port_in_use:
        print("[WARN] Port 8000 is already in use. Attempting to free it...",
              file=sys.stderr, flush=True)
        _kill_process_on_port(8000)
        # Re-check
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.settimeout(1)
            probe.connect(("localhost", 8000))
            probe.close()
            still_in_use = True
        except Exception:
            still_in_use = False

        if still_in_use:
            raise RuntimeError(
                "Port 8000 is occupied and could not be freed. "
                "Stop the conflicting process before running inference."
            )

    # ── Guard: image name must be non-empty ───────────────────────────────
    if not image_name or not image_name.strip():
        raise ValueError("IMAGE_NAME is empty or None; cannot start Docker container.")

    # ── Attempt docker run ─────────────────────────────────────────────────
    try:
        container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", "8000:8000", image_name],
            stderr=subprocess.PIPE,
        ).decode().strip()
    except subprocess.CalledProcessError as exc:
        stderr_msg = exc.stderr.decode().strip() if exc.stderr else ""
        raise RuntimeError(
            f"'docker run' failed (exit {exc.returncode}): {stderr_msg}"
        ) from exc
    except FileNotFoundError:
        raise RuntimeError(
            "Docker executable not found. Make sure Docker is installed and on PATH."
        )

    if not container_id:
        raise RuntimeError("'docker run' returned an empty container ID.")

    print(f"[INFO] Container: {container_id[:12]}", file=sys.stderr, flush=True)
    return container_id


def wait_for_server(url: str, retries: int = 30) -> None:
    """Block until /health returns 200 or raise on timeout."""
    print("[INFO] Waiting for server...", file=sys.stderr, flush=True)
    last_exc: Optional[Exception] = None
    for _ in range(retries):
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                print("[INFO] Server is ready.", file=sys.stderr, flush=True)
                return
            # Non-200 but server responded — keep waiting
        except httpx.TransportError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        time.sleep(1)
    raise RuntimeError(
        f"Server at {url} did not become ready after {retries}s. "
        f"Last error: {last_exc}"
    )


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

        # Guard: completion or its nested fields may be None
        if (
            not completion
            or not completion.choices
            or completion.choices[0].message is None
        ):
            print("[WARN] LLM returned empty completion; using fallback.", file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-empty-completion"}

        text = (completion.choices[0].message.content or "").strip()

        if not text:
            print("[WARN] LLM returned empty content; using fallback.", file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-empty-content"}

        # Strip markdown code fences if model wraps response in ```json ... ```
        if text.startswith("```"):
            parts = text.split("```")
            # parts[1] is the content inside the fences
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        # Guard: text may still not be valid JSON after stripping
        if not text:
            print("[WARN] LLM text empty after fence-strip; using fallback.", file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-empty-after-strip"}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as jexc:
            print(f"[WARN] JSON parse failed ({jexc}); raw text: {text!r}; using fallback.",
                  file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-json-error"}

        # Guard: parsed must be a dict
        if not isinstance(parsed, dict):
            print(f"[WARN] LLM JSON is not a dict ({type(parsed).__name__}); using fallback.",
                  file=sys.stderr, flush=True)
            return {"action_type": fallback_action, "rationale": "fallback-not-dict"}

        action_type = parsed.get("action", fallback_action)
        rationale   = parsed.get("reasoning", "")

        # Guard: action_type must be a non-empty string
        if not isinstance(action_type, str) or not action_type.strip():
            action_type = fallback_action
        if not isinstance(rationale, str):
            rationale = str(rationale)

        return {
            "action_type": action_type,
            "rationale":   rationale,
        }

    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", file=sys.stderr, flush=True)
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
            # Guard: env.reset() may raise or return None
            try:
                result = env.reset()
            except Exception as exc:
                print(f"[DEBUG] env.reset() failed: {exc}", file=sys.stderr, flush=True)
                raise

            if result is None:
                raise RuntimeError("env.reset() returned None; cannot proceed.")

            obs  = result.observation    # ApplicantObservation (Phase 1 start)
            done = result.done

            if obs is None:
                raise RuntimeError("env.reset() returned result with None observation.")

            for step in range(1, max_steps + 1):
                if done:
                    break

                # Guard: LLM call is already wrapped internally; always returns a dict
                action_dict = get_llm_action(llm_client, obs, step)
                action_type = action_dict.get("action_type", "APPROVE")
                rationale   = action_dict.get("rationale", "")

                # Guard: ensure action_type and rationale are plain strings
                if not isinstance(action_type, str) or not action_type.strip():
                    action_type = (
                        "APPROVE" if isinstance(obs, ApplicantObservation) else "DO_NOTHING"
                    )
                if not isinstance(rationale, str):
                    rationale = str(rationale)

                # CreditAction field names confirmed from models.py:
                #   action_type: ActionType  (the action string)
                #   rationale:   str         (the reasoning)
                try:
                    result = env.step(CreditAction(
                        action_type=action_type,
                        rationale=rationale,
                    ))
                except Exception as exc:
                    print(f"[DEBUG] env.step() failed at step {step}: {exc}",
                          file=sys.stderr, flush=True)
                    raise

                if result is None:
                    print(f"[WARN] env.step() returned None at step {step}; ending episode.",
                          file=sys.stderr, flush=True)
                    break

                obs    = result.observation   # may switch to MonitoringObservation in Phase 2
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

                # last_action_result is a field on the Pydantic obs, not an error key
                # We only surface it when it looks like an error (non-empty, non-success text)
                last_result = ""
                if obs is not None:
                    try:
                        last_result = getattr(obs, "last_action_result", "") or ""
                    except Exception:
                        last_result = ""
                error = last_result if last_result else None

                rewards.append(reward)
                steps_taken = step

                try:
                    action_str = json.dumps(
                        {"action": action_type, "reasoning": rationale},
                        ensure_ascii=False,
                    )
                except (TypeError, ValueError):
                    action_str = json.dumps(
                        {"action": str(action_type), "reasoning": str(rationale)},
                        ensure_ascii=False,
                    )

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

                # Guard: if obs is None after a step, we cannot continue
                if obs is None:
                    print(f"[WARN] obs is None after step {step}; ending episode.",
                          file=sys.stderr, flush=True)
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

    # Guard: validate API_BASE_URL is a non-empty string
    if not API_BASE_URL or not API_BASE_URL.strip():
        print("[ERROR] API_BASE_URL is empty or not set.", flush=True)
        sys.exit(1)

    # Guard: initialise OpenAI client safely
    try:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[ERROR] Failed to initialise OpenAI client: {exc}", flush=True)
        sys.exit(1)

    print(f"[INFO] Model  : {MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[INFO] Image  : {IMAGE_NAME}", file=sys.stderr, flush=True)
    print(f"[INFO] Server : {SERVER_URL}", file=sys.stderr, flush=True)
    print(f"[INFO] Tasks  : {[t['name'] for t in TASKS]}", file=sys.stderr, flush=True)
    print("", file=sys.stderr, flush=True)

    # Guard: start_container may raise — catch and exit cleanly
    container_id = ""
    try:
        container_id = start_container(IMAGE_NAME)
    except Exception as exc:
        print(f"[ERROR] Could not start Docker container: {exc}", flush=True)
        sys.exit(1)

    try:
        # Guard: wait_for_server may raise — let it propagate so finally still cleans up
        try:
            wait_for_server(SERVER_URL)
        except RuntimeError as exc:
            print(f"[ERROR] Server never became ready: {exc}", flush=True)
            # Still run tasks loop so log_end is always emitted; tasks will fail gracefully
            # (env connections will raise and be caught inside run_task)

        scores = {}
        for task in TASKS:
            try:
                scores[task["name"]] = run_task(task, llm_client)
            except Exception as exc:
                # Individual task crash must not abort other tasks
                print(f"[DEBUG] Unhandled error in task '{task['name']}': {exc}",
                      file=sys.stderr, flush=True)
                scores[task["name"]] = 0.0
            print("", file=sys.stderr, flush=True)

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

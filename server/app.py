"""
server/app.py — FastAPI entry point for the Microfinance Credit Decision Environment.

The environment exposes two observation types across its two phases:
  Phase 1 (APPLICATION) → ApplicantObservation
  Phase 2 (MONITORING)  → MonitoringObservation

OpenEnv's create_fastapi_app only accepts one observation type for the /step
endpoint schema, so we register ApplicantObservation as the declared type
(it's the entry point every episode starts with) and let MonitoringObservation
pass through via the metadata dict — both are dataclasses that serialise
cleanly to JSON.

The /state endpoint always returns MicrofinanceState regardless of phase.
"""

import sys
import os

# Ensure both server/ dir and project root are on the path
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # server/
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from openenv.core.env_server import create_fastapi_app
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from models import CreditAction, ApplicantObservation
from server.microfinance_env_environment import MicrofinanceEnvironment, TASK_CONFIGS

# ── Instantiate environment ────────────────────────────────────────────────
# DATASET_SIZE, SEED, and TASK_NAME are read from env vars so the Dockerfile
# can override them without rebuilding the image.
_dataset_size = int(os.environ.get("DATASET_SIZE", "300"))
_seed         = int(os.environ.get("SEED", "42"))
_task_name    = os.environ.get("TASK_NAME", "basic_lending")

# create_fastapi_app expects a callable (factory).
# We create one instance and always return it so /reset and /step share state.
_env_instance = MicrofinanceEnvironment(
    dataset_size=_dataset_size,
    seed=_seed,
    task_name=_task_name
)

def _env_factory():
    return _env_instance

# ── Create the ASGI app ────────────────────────────────────────────────────
app = create_fastapi_app(_env_factory, CreditAction, ApplicantObservation)

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
else:
    print(f"[WARNING] Static directory not found: {_static_dir}")


# ── /set_task — dynamically switch task difficulty without restarting ──────
class SetTaskRequest(BaseModel):
    task_name: str

@app.post("/set_task", include_in_schema=True, tags=["environment"])
async def set_task(req: SetTaskRequest):
    """
    Switch the active task configuration (basic_lending / noisy_signals /
    adversarial_portfolio) at runtime.  Must be called BEFORE /reset so the
    next episode uses the requested difficulty.
    """
    if req.task_name not in TASK_CONFIGS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{req.task_name}'. Valid tasks: {list(TASK_CONFIGS.keys())}"
        )
    _env_instance.set_task(req.task_name)
    return {"status": "ok", "active_task": req.task_name}


# ── Serve dashboard UI ─────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Static UI not found. Check /server/static folder."}




@app.get("/web", include_in_schema=False)
async def web():
    return await root()


# ── Entry point for OpenEnv runner ─────────────────────────────────
def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
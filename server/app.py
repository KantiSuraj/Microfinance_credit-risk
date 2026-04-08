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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models import CreditAction, ApplicantObservation
from server.microfinance_env_environment import MicrofinanceEnvironment

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

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(os.path.join(_static_dir, "index.html"))
else:
    print(f"[WARNING] Static directory not found: {_static_dir}")

    @app.get("/", include_in_schema=False)
    async def root():
        return {"error": "Static UI not found. Check /server/static folder."}




# ── Entry point for OpenEnv runner ─────────────────────────────────
def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_override():
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return "<h1>UI not found</h1>"


@app.get("/web", include_in_schema=False)
async def web():
    return await root_override()


if __name__ == "__main__":
    main()

    
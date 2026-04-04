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

# Ensure the env root is on the path (mirrors PYTHONPATH in Dockerfile)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from models import CreditAction, ApplicantObservation
from microfinance_env_environment import MicrofinanceEnvironment

# ── Instantiate environment ────────────────────────────────────────────────
# DATASET_SIZE and SEED are read from env vars so the Dockerfile can
# override them without rebuilding the image.
_dataset_size = int(os.environ.get("DATASET_SIZE", "300"))
_seed         = int(os.environ.get("SEED", "42"))

env = MicrofinanceEnvironment(dataset_size=_dataset_size, seed=_seed)

# ── Create the ASGI app ────────────────────────────────────────────────────
app = create_fastapi_app(env, CreditAction, ApplicantObservation)
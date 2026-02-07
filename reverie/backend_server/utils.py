"""
Utilities and configuration for backend server integrations.

This shim provides OpenAI API key access expected by modules that import
`from utils import openai_api_key` as documented in the root README.

Secrets should come from environment variables (loaded automatically from the
repo-root `.env` if present). Do NOT hardcode secrets in this file.
"""

import os

# Ensure `.env` is loaded before we read env vars.
try:
    from reverie.backend_server.env import load_env as _load_env  # type: ignore
except Exception:
    _load_env = None

if _load_env:
    _load_env()

# Prefer environment variable; fallback to empty string (no secret in repo)
openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# Owner metadata (optional)
key_owner = os.environ.get("OPENAI_KEY_OWNER", "local")

# Paths used by the original environment; keep for compatibility if needed
maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose flag
debug = True





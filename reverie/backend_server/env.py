"""
Environment loading helpers.

We keep secrets out of the repo by using a local `.env` file at the repo root
(gitignored). This module auto-loads it (if present) so the rest of the code
can rely on standard environment variables like OPENAI_API_KEY.
"""

from __future__ import annotations

from pathlib import Path
import os


_DID_LOAD = False


def _repo_root() -> Path:
    # reverie/backend_server/env.py -> repo root is 2 parents up
    # (.../repo/reverie/backend_server/env.py)
    return Path(__file__).resolve().parents[2]


def load_env(*, override: bool = False) -> None:
    """
    Load environment variables from `<repo>/.env` if it exists.

    - Safe to call multiple times (no-op after the first call).
    - If `override=True`, values in `.env` overwrite already-set env vars.
    """
    global _DID_LOAD
    if _DID_LOAD:
        return

    dotenv_path = _repo_root() / ".env"
    if not dotenv_path.exists():
        _DID_LOAD = True
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        # If python-dotenv isn't installed, just proceed with existing env vars.
        _DID_LOAD = True
        return

    load_dotenv(dotenv_path, override=override)
    _DID_LOAD = True


def getenv(name: str, default: str | None = None) -> str | None:
    """
    Convenience: ensure `.env` is loaded before reading an env var.
    """
    load_env()
    return os.environ.get(name, default)



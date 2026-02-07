#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path


def main():
    # Load repo-root .env if present (same behavior as reverie/backend_server/env.py)
    try:
        from dotenv import load_dotenv  # type: ignore
        repo_root = Path(__file__).resolve().parents[2]
        dotenv_path = repo_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=False)
    except Exception:
        pass

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'frontend_server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

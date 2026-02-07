#!/usr/bin/env python3
"""
Headless runner for the classic Reverie tile-world backend.

This is suitable for `scripts/run_stack.sh`:
- Creates a new sim by forking an existing base sim in storage
- Writes temp_storage/curr_sim_code.json and curr_step.json (used by Django UI)
- Runs the classic backend loop (waits for environment/<step>.json, then writes movement/<step>.json)

Hybrid mode:
- Set REVERIE_AGENT_MODE=hybrid to use HybridPersona (classic + predictive overlay + schedule tools)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path
import sys


def _default_sim_code(prefix: str = "hybrid") -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def main() -> None:
    ap = argparse.ArgumentParser()
    # Default to a smaller baseline to keep startup fast.
    # You can override with env FORK_SIM_CODE=... or --fork ...
    ap.add_argument("--fork", dest="fork_sim_code", default=os.environ.get("FORK_SIM_CODE", "enneagram9_smoketest"))
    ap.add_argument("--sim", dest="sim_code", default=os.environ.get("SIM_CODE") or _default_sim_code("hybrid"))
    ap.add_argument("--steps", dest="steps", type=int, default=int(os.environ.get("SIM_STEPS", "-1")))
    ap.add_argument("--dry-run", action="store_true", help="Print resolved config and exit.")
    args = ap.parse_args()

    # Ensure we run relative-path expectations from backend_server directory.
    backend_server_dir = Path(__file__).resolve().parents[1]
    os.chdir(str(backend_server_dir))
    # When running `python scripts/run_reverie_headless.py`, sys.path[0] is the
    # scripts/ directory, not backend_server/. Add backend_server so `import reverie`
    # (reverie.py) and sibling modules (env.py, utils.py, etc.) resolve correctly.
    if str(backend_server_dir) not in sys.path:
        sys.path.insert(0, str(backend_server_dir))

    # Import after chdir so relative fs_storage/fs_temp_storage in utils.py resolve correctly.
    from reverie import ReverieServer  # type: ignore

    if args.dry_run:
        print(f"backend_server_dir={backend_server_dir}")
        print(f"REVERIE_AGENT_MODE={os.environ.get('REVERIE_AGENT_MODE','classic')}")
        print(f"fork_sim_code={args.fork_sim_code}")
        print(f"sim_code={args.sim_code}")
        print(f"steps={args.steps}")
        return

    rs = ReverieServer(args.fork_sim_code, args.sim_code)
    rs.start_server(args.steps)


if __name__ == "__main__":
    main()



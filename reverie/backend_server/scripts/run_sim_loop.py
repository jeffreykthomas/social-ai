import asyncio
import sys
from pathlib import Path


async def main() -> None:
    # Ensure backend_server is on sys.path so `persona.*` imports work.
    backend_server_dir = Path(__file__).resolve().parents[1]
    if str(backend_server_dir) not in sys.path:
        sys.path.insert(0, str(backend_server_dir))

    # Load .env (repo root)
    try:
        from env import load_env  # type: ignore

        load_env()
    except Exception:
        pass

    from persona.persona_manager import get_manager  # type: ignore

    m = get_manager(use_predictive=True)

    # If agents already exist (e.g. hot-reload), don't recreate.
    if not m.agents:
        m.create_agent("Isabella Rodriguez", profile_id="enneagram_9")
        m.create_agent("Maria Lopez", profile_id="enneagram_2")

    while True:
        await m.update_agents()
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())



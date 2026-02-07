import os
import sys
import json
import requests
from pathlib import Path


def main() -> int:
    # Ensure repo root is on sys.path when running this script by file path.
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from reverie.backend_server.env import load_env  # type: ignore
        load_env()
    except Exception:
        pass

    base_url = os.environ.get("STUDENT_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    api_key = os.environ.get("STUDENT_API_KEY", "local-dev")
    model = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-32B-Instruct")

    headers = {"Authorization": f"Bearer {api_key}"}

    r = requests.get(f"{base_url}/models", headers=headers, timeout=10)
    print("GET /models:", r.status_code)
    print(r.text[:500])
    r.raise_for_status()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
    print("POST /chat/completions:", r.status_code)
    print(r.text[:800])
    r.raise_for_status()

    data = r.json()
    msg = data["choices"][0]["message"]
    print("assistant_message:", msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




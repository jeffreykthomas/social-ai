import os
import sys
import json
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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in environment.")
        return 2

    # Late imports so the message is clearer if deps are missing.
    from openai import OpenAI  # type: ignore

    # Use the same router as the simulation to validate integration + logging.
    from reverie.backend_server.models.router import ModelRouter
    from reverie.backend_server.models.providers import extract_tool_call

    client = OpenAI(api_key=api_key)
    router = ModelRouter(openai_client=client)

    print("Teacher smoke test: text generation")
    msg, debug = router.chat(
        provider="teacher",
        task="externalize",
        messages=[
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "Say OK."},
        ],
        temperature=0.0,
        max_tokens=32,
    )
    print("assistant_message:", msg)
    print("usage:", (debug or {}).get("usage"))
    if (msg.get("content") or "").strip() != "OK":
        print("WARNING: expected 'OK'")

    print("\nTeacher smoke test: tool call (forced)")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "do_nothing",
                "description": "Take no external action.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    msg2, debug2 = router.chat(
        provider="teacher",
        task="action",
        messages=[
            {"role": "system", "content": "Call the tool exactly once."},
            {"role": "user", "content": "Call do_nothing now."},
        ],
        temperature=0.0,
        max_tokens=128,
        tools=tools,
        tool_choice="required",
    )
    print("assistant_message:", msg2)
    print("usage:", (debug2 or {}).get("usage"))

    tc = extract_tool_call(msg2)
    print("parsed_tool_call:", tc)
    if not tc:
        print("WARNING: no tool call parsed (check tool schema compatibility).")

    print("\nOK: teacher reachable via Responses API through ModelRouter.")
    print("Logs (if enabled): reverie/backend_server/distill_logs/teacher.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



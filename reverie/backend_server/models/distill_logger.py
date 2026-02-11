import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class DistillLogger:
    """Append-only JSONL logger for teacher/student interactions."""

    def __init__(self, output_dir: str, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled

    def log(
        self,
        *,
        provider: str,
        task: str,
        model: str,
        messages: Any,
        response: Any,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
        filename: str = "distill.jsonl",
    ) -> None:
        if not self.enabled:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)

        record: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "provider": provider,
            "task": task,
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "response": response,
        }
        if meta:
            record["meta"] = meta

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



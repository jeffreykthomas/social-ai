"""Utilities for capturing agent experiences for downstream fine-tuning."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import threading


class EpisodeLogger:
    """Append-only JSONL logger for agent trajectories."""

    def __init__(self,
                 base_dir: Optional[Path] = None,
                 buffer_size: int = 20,
                 session_id: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent / "logs" / "episodes"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self._buffer: list[Dict[str, Any]] = []
        self._lock = threading.Lock()

        self.session_id = session_id or datetime.utcnow().strftime("session_%Y%m%d_%H%M%S")
        self._active_path = self.base_dir / f"{self.session_id}.jsonl"

    # ------------------------------------------------------------------
    # Session management

    def start_session(self, session_id: str) -> None:
        """Rotate to a new JSONL file for a fresh simulation episode."""
        with self._lock:
            self.flush()
            self.session_id = session_id
            self._active_path = self.base_dir / f"{self.session_id}.jsonl"

    # ------------------------------------------------------------------
    # Logging helpers

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Record a generic event with automatic timestamping."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            **payload,
        }
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.buffer_size:
                self._flush_locked()

    def log_speech(self,
                   agent_name: str,
                   content: str,
                   needs: Dict[str, float],
                   location: Optional[str] = None,
                   monologue: Optional[Iterable[str]] = None,
                   nearby_agents: Optional[Iterable[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "agent": agent_name,
            "content": content,
            "needs": needs,
        }
        if location is not None:
            payload["location"] = location
        if monologue is not None:
            payload["thoughts"] = list(monologue)
        if nearby_agents is not None:
            payload["nearby_agents"] = list(nearby_agents)
        if metadata:
            payload["metadata"] = metadata
        self.log_event("speech", payload)

    def log_action(self,
                   agent_name: str,
                   action: Dict[str, Any],
                   needs: Dict[str, float],
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "agent": agent_name,
            "action": action,
            "needs": needs,
        }
        if metadata:
            payload["metadata"] = metadata
        self.log_event("action", payload)

    def log_reward(self,
                   agent_name: str,
                   reward: float,
                   needs: Dict[str, float],
                   reason: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {
            "agent": agent_name,
            "reward": reward,
            "needs": needs,
        }
        if reason:
            payload["reason"] = reason
        self.log_event("reward", payload)

    # ------------------------------------------------------------------
    # Flush helpers

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        with self._active_path.open("a", encoding="utf-8") as handle:
            for record in self._buffer:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._buffer.clear()

    def close(self) -> None:
        self.flush()


__all__ = ["EpisodeLogger"]

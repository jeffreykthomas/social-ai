import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .distill_logger import DistillLogger
from .providers import OpenAICompatHTTPProvider, OpenAIResponsesProvider


class ModelRouter:
    """
    Routes LLM calls to teacher (OpenAI) or student (local) based on task/provider config.
    """

    def __init__(self, config_path: Optional[str] = None, openai_client=None):
        if config_path is None:
            config_path = str(Path(__file__).parent / "model_config.yaml")
        self.config_path = config_path
        self.config = self._load_config(config_path)

        distill_cfg = self.config.get("distillation", {})
        # Resolve output_dir relative to the *config file* (not the process CWD)
        # to avoid double-nesting when the sim chdirs into backend_server/.
        raw_dir = distill_cfg.get("output_dir", "distill_logs")
        if not os.path.isabs(raw_dir):
            raw_dir = str(Path(config_path).resolve().parent.parent / raw_dir)
        self.distill = DistillLogger(
            output_dir=raw_dir,
            enabled=bool(distill_cfg.get("enabled", True)),
        )
        self.log_teacher = bool(distill_cfg.get("log_teacher", True))
        self.log_student = bool(distill_cfg.get("log_student", False))

        teacher_cfg = self.config.get("teacher", {})
        student_cfg = self.config.get("student", {})

        self.teacher_models = teacher_cfg.get("models", {})
        self.teacher_default_model = teacher_cfg.get("default_model", "gpt-4o-mini")

        if openai_client is None:
            raise ValueError("openai_client is required for OpenAIResponsesProvider")
        self.teacher = OpenAIResponsesProvider(openai_client, default_model=self.teacher_default_model)

        self.student = OpenAICompatHTTPProvider(
            base_url=student_cfg.get("base_url", "http://localhost:8001/v1"),
            api_key=student_cfg.get("api_key", "local-dev"),
            model=student_cfg.get("model", "Qwen/Qwen2.5-32B-Instruct"),
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def teacher_model_for_task(self, task: str) -> str:
        return self.teacher_models.get(task, self.teacher_default_model)

    def provider_for_task(self, task: str) -> str:
        """
        Returns "teacher" or "student" for a task.

        Override with env:
          - MODEL_PROVIDER=teacher|student (forces all tasks)
        """
        forced = os.environ.get("MODEL_PROVIDER")
        if forced in ("teacher", "student"):
            return forced

        runtime = self.config.get("runtime", {}) or {}
        mapping = runtime.get("task_providers", {}) or {}
        provider = mapping.get(task, "teacher")
        return provider if provider in ("teacher", "student") else "teacher"

    def chat(
        self,
        *,
        provider: str,
        task: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        provider: "teacher" | "student"
        returns: (message, debug)
        """
        if provider not in ("teacher", "student"):
            raise ValueError(f"Unknown provider: {provider}")

        if provider == "teacher":
            model = self.teacher_model_for_task(task)
            msg, debug = self.teacher.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
            )
            if self.log_teacher:
                self.distill.log(
                    provider="teacher",
                    task=task,
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    response=msg,
                    meta={"usage": debug.get("usage"), **(meta or {})},
                    filename="teacher.jsonl",
                )
            return msg, debug

        # student
        model = self.config.get("student", {}).get("model")
        msg, debug = self.student.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        if self.log_student:
            self.distill.log(
                provider="student",
                task=task,
                model=model or "",
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                response=msg,
                meta={"usage": debug.get("usage"), **(meta or {})},
                filename="student.jsonl",
            )
        return msg, debug


_router_instance: Optional[ModelRouter] = None


def get_router(openai_client=None) -> ModelRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter(openai_client=openai_client)
    return _router_instance



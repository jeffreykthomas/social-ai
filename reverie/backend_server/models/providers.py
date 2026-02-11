import json
from typing import Any, Dict, List, Optional, Tuple

import requests


class OpenAIResponsesProvider:
    """
    Teacher provider via the modern OpenAI Python SDK (Responses API).

    Expects a client compatible with:
      client.responses.create(model=..., input=[...], tools=[...], tool_choice=..., ...)
    """

    def __init__(self, client: Any, default_model: str):
        self.client = client
        self.default_model = default_model

    def _to_dict(self, resp: Any) -> Dict[str, Any]:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
        # Last resort
        return json.loads(json.dumps(resp, default=str))

    def chat(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        use_model = model or self.default_model

        # Responses API tool schema differs from Chat Completions:
        # - Responses: {"type":"function","name":...,"description":...,"parameters":...}
        # - ChatCompletions: {"type":"function","function":{...}}
        tools_for_responses = None
        if tools is not None:
            tools_for_responses = []
            for t in tools:
                if not isinstance(t, dict):
                    continue
                if t.get("type") == "function" and isinstance(t.get("function"), dict):
                    fn = t["function"]
                    tools_for_responses.append(
                        {
                            "type": "function",
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "parameters": fn.get("parameters"),
                        }
                    )
                else:
                    tools_for_responses.append(t)

        tool_choice_for_responses = tool_choice
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
            tool_choice_for_responses = {"type": "function", "name": tool_choice["function"].get("name")}

        kwargs: Dict[str, Any] = {
            "model": use_model,
            "input": messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if tools_for_responses is not None:
            kwargs["tools"] = tools_for_responses
        if tool_choice_for_responses is not None:
            kwargs["tool_choice"] = tool_choice_for_responses

        resp = self.client.responses.create(**kwargs)
        d = self._to_dict(resp)

        # Text extraction
        text = getattr(resp, "output_text", None) or d.get("output_text") or ""

        # Tool calls extraction (normalize to Chat Completions style)
        tool_calls: List[Dict[str, Any]] = []
        for item in d.get("output", []) or []:
            # Responses API emits function calls as output items
            if isinstance(item, dict) and item.get("type") == "function_call":
                name = item.get("name") or ""
                args = item.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": item.get("call_id") or item.get("id"),
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                )

        msg: Dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls

        usage = d.get("usage", {}) or {}
        return msg, {"usage": usage, "raw": d}


class OpenAICompatHTTPProvider:
    """Student provider via an OpenAI-compatible HTTP server (e.g., vLLM)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        use_model = model or self.model
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload: Dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        r = requests.post(url, headers=headers, json=payload, timeout=120)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # Include server error body to make debugging vLLM/tool-calling much easier.
            raise requests.HTTPError(f"{e} | body={r.text[:2000]}") from None
        data = r.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        return msg, {"usage": usage, "raw": data}


def extract_tool_call(message: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Normalize tool/function calls across providers.

    Returns (tool_name, arguments_dict) or None.
    """
    # OpenAI old format: {"function_call": {"name": "...", "arguments": "...json..."}}
    if "function_call" in message and message["function_call"]:
        fc = message["function_call"]
        name = fc.get("name")
        args_raw = fc.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception:
            args = {}
        if name:
            return name, args

    # Newer tool-calls: {"tool_calls": [{"function": {"name":..., "arguments":"{...}"}}]}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0]
        fn = (tc.get("function") or {})
        name = fn.get("name")
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception:
            args = {}
        if name:
            return name, args

    return None



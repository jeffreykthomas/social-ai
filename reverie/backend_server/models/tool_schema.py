from typing import Any, Dict, List


def agent_action_tools() -> List[Dict[str, Any]]:
    """
    Tools for action selection.

    This is intentionally small to make early distillation stable.
    We can expand later (schedules, notes, transactions).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "initiate_conversation",
                "description": "Start a conversation with a nearby agent to increase connection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_name": {
                            "type": "string",
                            "description": "Name of the agent to talk to.",
                        }
                    },
                    "required": ["target_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_for_help",
                "description": "Ask nearby agents for help (safety/support).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "What you need help with (short).",
                        }
                    },
                    "required": ["topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "offer_help",
                "description": "Offer help to a nearby agent who seems in need.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_name": {"type": "string"},
                        "topic": {"type": "string"},
                    },
                    "required": ["target_name", "topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "do_nothing",
                "description": "Take no external action this step.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def agent_schedule_tools() -> List[Dict[str, Any]]:
    """
    Tools for daily schedule creation.

    This is intended to be called when an agent has no schedule or the schedule is stale.
    The manager can enforce this by setting tool_choice="required".
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "set_daily_schedule",
                "description": "Create a schedule for the agent's next waking block (e.g., 8am–8pm).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "Ordered schedule items. Durations should total ~720 minutes for a 12-hour day.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "activity": {"type": "string", "description": "What the agent will do."},
                                    "location": {"type": "string", "description": "Canonical location name (choose from allowed list)."},
                                    "duration_minutes": {"type": "integer", "description": "How long, in minutes."},
                                    "social_intent": {"type": "string", "description": "Optional: who/what they hope to interact with."},
                                },
                                "required": ["activity", "location", "duration_minutes"],
                            },
                        }
                    },
                    "required": ["items"],
                },
            },
        }
    ]


def agent_schedule_adjust_tools() -> List[Dict[str, Any]]:
    """
    Tools to choose what to do with the schedule after an interaction.
    Exactly one should be chosen:
      - stick_with_schedule: do nothing
      - edit_schedule: rewrite the remaining schedule items for the current day
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "stick_with_schedule",
                "description": "Keep the current schedule unchanged.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_schedule",
                "description": "Rewrite the remaining schedule items for today (from 'now' until end of waking window).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "Ordered schedule items for the remaining time window.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "activity": {"type": "string"},
                                    "location": {"type": "string"},
                                    "duration_minutes": {"type": "integer"},
                                    "social_intent": {"type": "string"},
                                },
                                "required": ["activity", "location", "duration_minutes"],
                            },
                        }
                    },
                    "required": ["items"],
                },
            },
        },
    ]


def tool_call_to_schedule_adjust(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "stick_with_schedule":
        return {"type": "stick_with_schedule"}
    if tool_name == "edit_schedule":
        return {"type": "edit_schedule", "items": arguments.get("items", [])}
    return {"type": "stick_with_schedule"}


def agent_commitment_tools() -> List[Dict[str, Any]]:
    """
    Tools for creating a future commitment (one-off or recurring).
    The model may also choose to do nothing.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "no_new_commitment",
                "description": "No new commitment should be created from this interaction.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_commitment",
                "description": "Create a new commitment the agent intends to keep (meeting, event, recurring work block).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "commitment": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique id; if omitted, manager will generate one."},
                                "title": {"type": "string", "description": "Short title, e.g., 'work shift' or 'Lakers game'."},
                                "location": {"type": "string", "description": "Canonical location name (choose from allowed list)."},
                                "participants": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Other agents who are expecting this agent (optional).",
                                },
                                "start_iso": {"type": "string", "description": "One-off start time in ISO format (optional)."},
                                "end_iso": {"type": "string", "description": "One-off end time in ISO format (optional)."},
                                "recurrence": {
                                    "type": "object",
                                    "description": "Recurring pattern (optional).",
                                    "properties": {
                                        "freq": {"type": "string", "description": "daily|weekly"},
                                        "time_hm": {"type": "string", "description": "HH:MM (24h) local sim time"},
                                        "days_of_week": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "description": "0=Mon ... 6=Sun (for weekly)",
                                        },
                                    },
                                },
                            },
                            "required": ["title", "location"],
                        }
                    },
                    "required": ["commitment"],
                },
            },
        },
    ]


def tool_call_to_commitment_action(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "no_new_commitment":
        return {"type": "no_new_commitment"}
    if tool_name == "add_commitment":
        return {"type": "add_commitment", "commitment": arguments.get("commitment", {})}
    return {"type": "no_new_commitment"}


def tool_call_to_schedule(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize schedule tool calls.
    """
    if tool_name == "set_daily_schedule":
        return {"type": "set_daily_schedule", "items": arguments.get("items", [])}
    return {"type": "set_daily_schedule", "items": []}


def tool_call_to_action(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a tool call into the action dict expected by PersonaManager."""
    if tool_name == "initiate_conversation":
        return {"type": "initiate_conversation", "target_name": arguments.get("target_name")}
    if tool_name == "ask_for_help":
        return {"type": "ask_for_help", "topic": arguments.get("topic")}
    if tool_name == "offer_help":
        return {
            "type": "offer_help",
            "target_name": arguments.get("target_name"),
            "topic": arguments.get("topic"),
        }
    return {"type": "do_nothing"}



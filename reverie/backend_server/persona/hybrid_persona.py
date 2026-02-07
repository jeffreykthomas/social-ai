"""
HybridPersona: Classic Reverie Persona + predictive/need-based overlay.

Goal:
- Keep classic tile-world behavior (perceive/retrieve/plan/execute, conversations, movement JSON)
- Add a per-step predictive tick (needs, inner monologue, lightweight predictions)
- Allow *tool-call-based* rescheduling by editing the remaining portion of scratch.f_daily_schedule

This is intentionally minimal: the classic planner still chooses actions; the overlay is for
monitoring + schedule adjustment hooks.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

from persona.persona import Persona


# Predictive overlay components (reused from predictive_persona)
try:
    from persona.predictive_persona import NeedState, PredictionModel, InternalMonologue  # type: ignore
except Exception:  # pragma: no cover
    NeedState = None  # type: ignore
    PredictionModel = None  # type: ignore
    InternalMonologue = None  # type: ignore


class HybridPersona(Persona):
    """
    A classic Persona with a lightweight predictive overlay.
    """

    def __init__(self, name: str, folder_mem_saved: str, *, enable_reschedule_tools: bool = True):
        super().__init__(name, folder_mem_saved)

        self.enable_reschedule_tools = enable_reschedule_tools

        # Overlay state lives primarily in Scratch for persistence.
        # We still keep small helper objects in-memory for convenience.
        self._overlay_enabled = (NeedState is not None and PredictionModel is not None and InternalMonologue is not None)

        if self._overlay_enabled:
            self._needs = NeedState()  # type: ignore[call-arg]
            # Restore needs if saved previously
            if isinstance(getattr(self.scratch, "need_states", None), dict) and self.scratch.need_states:
                try:
                    for k, v in self.scratch.need_states.items():
                        if k in self._needs.needs:
                            self._needs.needs[k] = float(v)
                except Exception:
                    pass

            self._pred = PredictionModel()  # type: ignore[call-arg]
            self._mono = InternalMonologue(self.name)  # type: ignore[call-arg]

        # Used to detect “conversation ended” boundary without persisting extra scratch keys.
        self._prev_chatting_with: Optional[str] = None

    # -------------------------
    # Classic loop integration
    # -------------------------

    def move(self, maze, personas, curr_tile, curr_time):
        """
        Same signature as Persona.move; we inject the overlay before/after planning.
        """
        # Update scratch with position/time (same as classic Persona)
        self.scratch.curr_tile = curr_tile

        new_day = False
        if not self.scratch.curr_time:
            new_day = "First day"
        elif (self.scratch.curr_time.strftime('%A %B %d') != curr_time.strftime('%A %B %d')):
            new_day = "New day"

        self._prev_chatting_with = getattr(self.scratch, "chatting_with", None)
        self.scratch.curr_time = curr_time

        # Classic cognition
        perceived = self.perceive(maze)
        retrieved = self.retrieve(perceived)

        # Predictive overlay tick (does not change classic action selection)
        self._hybrid_predictive_tick(perceived=perceived, retrieved=retrieved, personas=personas)

        plan = self.plan(maze, personas, new_day, retrieved)
        self.reflect()

        # Tool-call schedule adjustment hook (edits remaining scratch.f_daily_schedule)
        self._maybe_adjust_schedule(new_day=new_day)

        return self.execute(maze, personas, plan)

    # -------------------------
    # Overlay: needs/monologue
    # -------------------------

    def _append_monologue(self, thought_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            ts = self.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S") if self.scratch.curr_time else None
        except Exception:
            ts = None
        entry = {"timestamp": ts, "type": thought_type, "content": content}
        if metadata:
            entry["metadata"] = metadata

        try:
            if not isinstance(self.scratch.internal_monologue, list):
                self.scratch.internal_monologue = []
            self.scratch.internal_monologue.append(entry)
            # Keep the stream bounded
            if len(self.scratch.internal_monologue) > 200:
                self.scratch.internal_monologue = self.scratch.internal_monologue[-200:]
        except Exception:
            pass

    def _hybrid_predictive_tick(self, *, perceived: Any, retrieved: Any, personas: Dict[str, Any]) -> None:
        """
        Update need states and internal monologue based on what was perceived/retrieved.
        """
        if not self._overlay_enabled:
            return

        # 1) Need decay (simple: 1 decay per sim step)
        try:
            self._needs.decay()
        except Exception:
            pass

        # 2) Record needs to scratch (persistence)
        try:
            self.scratch.need_states = dict(getattr(self._needs, "needs", {}) or {})
        except Exception:
            self.scratch.need_states = {}

        # 3) Maintain a small “recent event description” buffer for prediction context
        try:
            rec = self.scratch.recent_event_descriptions
            if not isinstance(rec, list):
                rec = []
            # Best-effort: treat perceived as iterable of nodes with .description
            for node in (perceived or []):
                desc = getattr(node, "description", None)
                if isinstance(desc, str) and desc:
                    rec.append(desc)
            # Also include who we're chatting with (if any) as a coarse “event”
            if getattr(self.scratch, "chatting_with", None):
                rec.append(f"chatting_with:{self.scratch.chatting_with}")
            rec = rec[-30:]
            self.scratch.recent_event_descriptions = rec
        except Exception:
            pass

        # 4) Monologue: highlight deficient needs
        try:
            needs = getattr(self._needs, "needs", {}) or {}
            if isinstance(needs, dict) and needs:
                lowest = sorted(needs.items(), key=lambda kv: kv[1])[:2]
                for need, value in lowest:
                    if value < 0.35:
                        self._append_monologue("need_awareness", f"My {need} feels low ({value:.2f})", {"need": need, "value": value})
        except Exception:
            pass

        # 5) Lightweight predictions: markov-ish prediction model, if it has vocabulary
        try:
            context = {"recent_events": list(self.scratch.recent_event_descriptions or [])}
            preds = self._pred.predict_next_events(context) or {}
            top = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)[:5]
            buf = [{"event": e, "probability": float(p)} for e, p in top]
            self.scratch.prediction_buffer = buf
            if top:
                e, p = top[0]
                self._append_monologue("prediction", f"I predict {e} might happen soon ({p:.0%})", {"event": e, "probability": p})
        except Exception:
            pass

    # -------------------------
    # Overlay: reschedule tools
    # -------------------------

    def _maybe_adjust_schedule(self, *, new_day: Any) -> None:
        """
        After meaningful boundaries, allow the agent to tool-call a schedule adjustment.

        Trigger policy (minimal):
        - do nothing if disabled or no router configured
        - consider adjustment if:
          - it's a new day, OR
          - a conversation just ended, OR
          - any need is critical (< 0.20)
        Gated to at most once per 3 in-sim hours.
        """
        if not self.enable_reschedule_tools:
            return
        if not self._overlay_enabled:
            return

        now = getattr(self.scratch, "curr_time", None)
        if not isinstance(now, datetime.datetime):
            return

        # Gate by time
        last_s = getattr(self.scratch, "last_schedule_adjust_time", None)
        last: Optional[datetime.datetime] = None
        if isinstance(last_s, str) and last_s:
            try:
                last = datetime.datetime.strptime(last_s, "%B %d, %Y, %H:%M:%S")
            except Exception:
                last = None
        if last and (now - last) < datetime.timedelta(hours=3):
            return

        # Triggers
        convo_ended = (self._prev_chatting_with is not None and getattr(self.scratch, "chatting_with", None) is None)
        critical = False
        try:
            needs = getattr(self._needs, "needs", {}) or {}
            critical = any(float(v) < 0.20 for v in needs.values())
        except Exception:
            critical = False

        if not (new_day or convo_ended or critical):
            return

        decision = _toolcall_schedule_adjust_for_persona(self)
        if not decision:
            return

        if decision.get("type") == "edit_schedule":
            items = decision.get("items") or []
            _splice_classic_schedule(self, items)
            try:
                self.scratch.last_schedule_adjust_time = now.strftime("%B %d, %Y, %H:%M:%S")
            except Exception:
                self.scratch.last_schedule_adjust_time = None


def _get_allowed_locations_from_spatial_memory(persona: Persona) -> List[str]:
    """
    Derive a canonical list of locations from spatial memory if available.
    Fallback to an empty list (caller can handle).
    """
    try:
        tree = getattr(persona, "s_mem", None)
        world = getattr(tree, "tree", {}) if tree else {}
        ville = world.get("the Ville", {})
        if isinstance(ville, dict) and ville:
            return sorted([k for k in ville.keys() if k])
    except Exception:
        pass
    return []


def _toolcall_schedule_adjust_for_persona(persona: HybridPersona) -> Optional[Dict[str, Any]]:
    """
    Ask the configured model (teacher/student) to either stick with schedule or edit the remaining schedule.
    Converts the tool call into a normalized dict compatible with tool_call_to_schedule_adjust.
    """
    # Local imports so classic mode can still run if these deps are missing.
    try:
        from models.router import get_router  # type: ignore
        from models.providers import extract_tool_call  # type: ignore
        from models.tool_schema import agent_schedule_adjust_tools, tool_call_to_schedule_adjust  # type: ignore
    except Exception:
        return None

    # Optional OpenAI integration; degrade gracefully if not installed.
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        OpenAI = None  # type: ignore

    if OpenAI is None:
        return None

    # Build router (singleton)
    try:
        router = get_router(openai_client=OpenAI())
    except Exception:
        return None

    # Build prompt from classic schedule + predictive needs
    locs = _get_allowed_locations_from_spatial_memory(persona)
    sched_preview = []
    try:
        # Represent schedule as a compact list with cumulative “minute of day”
        t = 0
        for act, dur in list(getattr(persona.scratch, "f_daily_schedule", []) or [])[:20]:
            sched_preview.append({"t_end_min": t + int(dur), "activity": str(act)[:120], "duration_minutes": int(dur)})
            t += int(dur)
    except Exception:
        sched_preview = []

    tools = agent_schedule_adjust_tools()

    prompt = (
        "A simulation step boundary was reached. Decide whether to keep the schedule unchanged, "
        "or to edit the remaining schedule for today.\n\n"
        f"Agent: {persona.scratch.name}\n"
        f"Time: {persona.scratch.curr_time.strftime('%B %d, %Y, %H:%M:%S')}\n"
        f"Current activity: {persona.scratch.act_description}\n"
        f"Current location/address: {persona.scratch.act_address}\n"
        f"Need states: {getattr(persona.scratch, 'need_states', {})}\n"
        f"Recent monologue (last 5): {[t.get('content') for t in (persona.scratch.internal_monologue or [])[-5:]]}\n\n"
        f"Allowed locations (choose exact strings when you include a location): {locs}\n\n"
        f"Current schedule (preview): {sched_preview}\n\n"
        "If editing schedule: return items for the remaining time only.\n"
        "For each item, provide:\n"
        "- activity: a short description\n"
        "- location: choose from allowed locations if possible; otherwise reuse current area\n"
        "- duration_minutes: integer\n"
        "Call exactly one tool: stick_with_schedule or edit_schedule.\n"
    )

    messages = [
        {"role": "system", "content": "Choose by calling one tool."},
        {"role": "user", "content": prompt},
    ]

    try:
        msg, _debug = router.chat(
            provider=router.provider_for_task("schedule"),
            task="schedule",
            messages=messages,
            temperature=0.3,
            max_tokens=450,
            tools=tools,
            tool_choice="required",
            meta={"agent": persona.scratch.name, "mode": "hybrid_classic"},
        )
        tc = extract_tool_call(msg)
        if not tc:
            return None
        tool_name, args = tc
        return tool_call_to_schedule_adjust(tool_name, args)
    except Exception:
        return None


def _splice_classic_schedule(persona: HybridPersona, items: List[Dict[str, Any]]) -> None:
    """
    Convert schedule items (activity/location/duration_minutes) into classic f_daily_schedule entries
    and splice them into the remaining part of the day.

    We embed location hints directly into the activity string so classic action->location prompts can use it.
    """
    try:
        now = persona.scratch.curr_time
        if not isinstance(now, datetime.datetime):
            return
        minutes_now = now.hour * 60 + now.minute

        # Find current index into f_daily_schedule
        idx = persona.scratch.get_f_daily_schedule_index()

        # Compute minutes covered by prefix [0:idx)
        prefix = []
        prefix_minutes = 0
        for act, dur in (persona.scratch.f_daily_schedule or [])[:idx]:
            prefix.append([act, int(dur)])
            prefix_minutes += int(dur)

        remaining_minutes = max(0, 1440 - prefix_minutes)
        if remaining_minutes <= 0:
            return

        # Normalize items into classic rows
        out_rows: List[List[Any]] = []
        used = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                dur = int(it.get("duration_minutes", 0))
            except Exception:
                dur = 0
            if dur <= 0:
                continue
            if used >= remaining_minutes:
                break
            dur = min(dur, remaining_minutes - used)
            act = str(it.get("activity", "")).strip()
            loc = str(it.get("location", "")).strip()
            if loc and act:
                # Location hint (classic code often splits on '@' already)
                act = f"{act} @ {loc}"
            elif loc and not act:
                act = f"go to {loc}"
            if not act:
                continue
            out_rows.append([act[:240], dur])
            used += dur

        # Pad remainder with sleeping if needed
        if used < remaining_minutes:
            out_rows.append(["sleeping", remaining_minutes - used])

        persona.scratch.f_daily_schedule = prefix + out_rows
        # Keep hourly_org in sync (classic code uses it for some prompts/decomposition decisions)
        persona.scratch.f_daily_schedule_hourly_org = persona.scratch.f_daily_schedule[:]
        # We also update the “daily plan requirement” string to reflect schedule change (best-effort)
        try:
            persona.scratch.daily_plan_req = f"(updated via schedule tool at {minutes_now} min) {persona.scratch.daily_plan_req}"
        except Exception:
            pass
    except Exception:
        return



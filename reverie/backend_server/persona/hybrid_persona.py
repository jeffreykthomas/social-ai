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

    def __init__(self, name: str, folder_mem_saved: str, *,
                 profile: Optional[Dict[str, Any]] = None,
                 enable_reschedule_tools: bool = True):
        super().__init__(name, folder_mem_saved)

        self.enable_reschedule_tools = enable_reschedule_tools

        # Enneagram / persona profile (dict from profiles/enneagram.yaml).
        self.profile = profile or {}
        self.persona_profile_id: Optional[str] = self.profile.get("id") if self.profile else None
        # Persist profile id on scratch so it survives save/load.
        if self.persona_profile_id:
            self.scratch.persona_profile_id = self.persona_profile_id

        # Overlay state lives primarily in Scratch for persistence.
        # We still keep small helper objects in-memory for convenience.
        self._overlay_enabled = (NeedState is not None and PredictionModel is not None and InternalMonologue is not None)

        if self._overlay_enabled:
            self._needs = NeedState(profile=self.profile)  # type: ignore[call-arg]
            # Restore needs if saved previously (overrides profile defaults
            # when resuming a sim that already ran some steps).
            if isinstance(getattr(self.scratch, "need_states", None), dict) and self.scratch.need_states:
                try:
                    for k, v in self.scratch.need_states.items():
                        if k in self._needs.needs:
                            self._needs.needs[k] = float(v)
                except Exception:
                    pass

            self._pred = PredictionModel()  # type: ignore[call-arg]
            self._mono = InternalMonologue(self.name)  # type: ignore[call-arg]

            # Pre-populate scratch.need_states so the monitor shows needs
            # even before the first sim step runs.
            try:
                self.scratch.need_states = dict(self._needs.needs)
            except Exception:
                pass

        # Used to detect “conversation ended” boundary without persisting extra scratch keys.
        self._prev_chatting_with: Optional[str] = None

        # Judge verdict ledger -- last N verdicts used as few-shot calibration
        # anchors so the judge model maintains a consistent numeric scale.
        self._judge_ledger: list = []
        self._last_judge_step: int = 0
        self._current_verdict = None
        self._step_counter: int = 0

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

    # Subjective thought templates keyed by need name.  Each list has
    # first-person thoughts that sound like an inner voice, not a status
    # report.  {name} = nearby agent name.
    _NEED_THOUGHTS = {
        "connection": [
            "I should reach out to someone today, maybe grab coffee together",
            "It's been a while since I had a real conversation with anyone",
            "I wonder what {name} is up to... might be nice to catch up",
            "I feel a bit isolated. I should make plans with someone",
        ],
        "safety": [
            "Something feels off today, I should stick to familiar places",
            "I need to double-check my schedule and make sure nothing falls through the cracks",
            "I'd feel better if I just stayed somewhere comfortable for a while",
        ],
        "approval": [
            "I wonder if people noticed the work I put in yesterday",
            "I should do something that shows what I'm capable of",
            "It would feel good to hear that someone appreciates what I do",
        ],
        "empathy": [
            "I wish someone really understood what I'm going through",
            "I should check in on the people around me, see how they're doing",
            "It's hard when no one asks how you're really feeling",
        ],
        "fun": [
            "Everything feels like a chore lately, I need to do something fun",
            "I should find something spontaneous to do today",
            "When was the last time I actually laughed? I need more of that",
        ],
        "attention": [
            "I feel kind of invisible today",
            "It would be nice if someone just acknowledged I was here",
            "Maybe I should speak up more in conversations",
        ],
        "achievement": [
            "I haven't accomplished anything meaningful lately, that bothers me",
            "I need to set a goal and actually follow through today",
            "I should focus on finishing something, even if it's small",
        ],
        "autonomy": [
            "I feel like I've been going along with everyone else's plans",
            "I need some time to myself to think",
            "I should do something on my own terms today",
        ],
        "purpose": [
            "What am I really doing with my time? I want it to matter",
            "I should focus on something that feels meaningful today",
            "I need to remind myself why I'm doing all this",
        ],
    }

    def _hybrid_predictive_tick(self, *, perceived: Any, retrieved: Any, personas: Dict[str, Any]) -> None:
        """
        Update need states and internal monologue based on what was perceived/retrieved.
        Skips decay and monologue while the agent is sleeping.
        """
        if not self._overlay_enabled:
            return

        import random as _rng

        # 1) Need decay -- skip while sleeping (needs are stable during rest)
        is_sleeping = "sleep" in (self.scratch.act_description or "").lower()
        if not is_sleeping:
            try:
                self._needs.decay()
            except Exception:
                pass

        # 1b) Activity-based need fulfillment via judge agent.
        #     Every ~30 steps (5 sim-minutes), the judge (GPT-5.2) evaluates
        #     the agent's current activity and returns a structured verdict
        #     with per-need impacts (positive and negative) and a narrative.
        #     Between judge calls, the most recent verdict's impacts are
        #     applied each step.  A semantic cache avoids re-evaluating
        #     similar activities.
        if not is_sleeping:
            try:
                from persona.judge import get_judge

                judge = get_judge()
                activity = self.scratch.act_description or ""
                location = self.scratch.act_address or self.scratch.living_area or ""
                chatting = getattr(self.scratch, "chatting_with", None)

                # Determine if it's time for a new judge evaluation (~every 30 steps)
                last_judge_step = getattr(self, "_last_judge_step", 0)
                last_verdict = getattr(self, "_current_verdict", None)
                current_step = getattr(self, "_step_counter", 0) + 1
                self._step_counter = current_step

                if current_step - last_judge_step >= 30 and activity:
                    # Gather context for the judge
                    nearby = [n for n in personas.keys() if n != self.name]
                    profile = getattr(self, "profile", {}) or {}
                    mono_entries = getattr(self.scratch, "internal_monologue", []) or []
                    recent_thoughts = [e.get("content", "") for e in mono_entries[-3:]]

                    verdict = judge.evaluate(
                        persona_name=self.name,
                        activity=activity,
                        location=location,
                        nearby_agents=nearby[:5],
                        current_needs=dict(self._needs.needs),
                        profile_label=profile.get("label", ""),
                        core_motive=profile.get("core_motive", ""),
                        recent_monologue=recent_thoughts,
                        chatting_with=chatting,
                        ledger=self._judge_ledger,
                    )

                    if verdict:
                        self._current_verdict = verdict
                        self._last_judge_step = current_step

                        # Append to the ledger for future calibration
                        self._judge_ledger.append({
                            "activity": activity[:80],
                            "magnitude": verdict.get("magnitude", ""),
                            "impacts": verdict.get("impacts", {}),
                            "sim_time": self.scratch.curr_time.strftime("%H:%M") if self.scratch.curr_time else "",
                        })
                        # Keep ledger bounded to 8 entries
                        if len(self._judge_ledger) > 8:
                            self._judge_ledger = self._judge_ledger[-8:]

                        # Feed the judge's narrative into the monologue stream
                        narrative = verdict.get("narrative", "")
                        if narrative:
                            self._append_monologue(
                                "observation", narrative,
                                {"source": "judge", "magnitude": verdict.get("magnitude", ""),
                                 "cached": verdict.get("cached", False)})

                # Apply the current verdict's impacts each step
                if getattr(self, "_current_verdict", None):
                    judge.apply_verdict(self._needs, self._current_verdict)

                # Baseline: chatting always gives a direct connection boost
                if chatting:
                    self._needs.update("connection", 0.003)
                    self._needs.update("attention", 0.002)
                    self._needs.update("empathy", 0.001)

            except Exception as _tick_err:
                # Fallback: tiny baseline fulfillment if judge is unavailable
                import sys
                print(f"[judge-tick] {self.name}: {_tick_err}", file=sys.stderr)
                try:
                    self._needs.update("safety", 0.0005)
                except Exception:
                    pass

        # 2) Record needs to scratch (persistence)
        try:
            self.scratch.need_states = dict(getattr(self._needs, "needs", {}) or {})
        except Exception:
            self.scratch.need_states = {}

        # 3) Maintain a small "recent event description" buffer for prediction context
        try:
            rec = self.scratch.recent_event_descriptions
            if not isinstance(rec, list):
                rec = []
            for node in (perceived or []):
                desc = getattr(node, "description", None)
                if isinstance(desc, str) and desc:
                    rec.append(desc)
            if getattr(self.scratch, "chatting_with", None):
                rec.append(f"chatting_with:{self.scratch.chatting_with}")
            rec = rec[-30:]
            self.scratch.recent_event_descriptions = rec
        except Exception:
            pass

        # Skip monologue and predictions while sleeping
        if is_sleeping:
            return

        # 4) Subjective monologue -- throttled to ~1 thought per 5 sim-minutes
        #    (30 steps at 10s/step) and voiced as first-person inner speech.
        try:
            mono = self.scratch.internal_monologue or []
            steps_since_last = 30  # default: eligible
            if mono:
                last_ts = mono[-1].get("timestamp")
                if last_ts and self.scratch.curr_time:
                    try:
                        last_dt = datetime.datetime.strptime(last_ts, "%B %d, %Y, %H:%M:%S")
                        steps_since_last = int((self.scratch.curr_time - last_dt).total_seconds() / max(1, 10))
                    except Exception:
                        pass

            if steps_since_last >= 30:
                needs = getattr(self._needs, "needs", {}) or {}
                if isinstance(needs, dict) and needs:
                    lowest = sorted(needs.items(), key=lambda kv: kv[1])
                    for need, value in lowest[:2]:
                        if value < 0.45:
                            templates = self._NEED_THOUGHTS.get(need, [])
                            if templates:
                                other_names = [n for n in personas.keys() if n != self.name]
                                other = _rng.choice(other_names) if other_names else "someone"
                                thought = _rng.choice(templates).format(name=other, value=value)
                                self._append_monologue("need_awareness", thought, {"need": need, "value": value})
                            break  # one thought per tick
        except Exception:
            pass

        # 5) Lightweight predictions
        try:
            context = {"recent_events": list(self.scratch.recent_event_descriptions or [])}
            preds = self._pred.predict_next_events(context) or {}
            top = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)[:5]
            buf = [{"event": e, "probability": float(p)} for e, p in top]
            self.scratch.prediction_buffer = buf
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



"""
Persona Manager for Predictive Need-Based Agents

This module manages the integration of predictive personas with the existing
Reverie system, handling agent creation, updates, and interactions.

Author: AI Playground Team
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
import numpy as np
import traceback
import os
from pathlib import Path

from persona.predictive_persona import PredictivePersona
from persona.cognitive_modules.need_aware_prompts import create_need_aware_prompt, validate_llm_response
from config.config_loader import get_config
from models.router import get_router
from models.providers import extract_tool_call
from models.tool_schema import (
    agent_action_tools,
    tool_call_to_action,
    agent_schedule_tools,
    tool_call_to_schedule,
    agent_schedule_adjust_tools,
    tool_call_to_schedule_adjust,
    agent_commitment_tools,
    tool_call_to_commitment_action,
)
from persona.profiles.profile_loader import get_profile, profile_style_text

# Local reasoning model
import sys
sys.path.append('../')
from llm_client import LLMConfig, get_client
from episode_logger import EpisodeLogger

# Optional OpenAI integration (teacher). If openai isn't installed, we degrade gracefully.
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


class PersonaManager:
    """
    Manages predictive personas and their interactions with the environment
    """
    
    def __init__(self, use_predictive: bool = True):
        """
        Initialize the persona manager
        
        Args:
            use_predictive: Whether to use predictive personas or original
        """
        self.use_predictive = use_predictive
        self.agents: Dict[str, Any] = {}  # agent_name -> agent instance
        self.config = get_config()

        llm_params = self.config.get_llm_params()
        if not llm_params.get('model_name'):
            raise ValueError("llm.model_name must be configured in need_config.yaml")

        self._llm_config = LLMConfig(
            model_name=llm_params['model_name'],
            device=llm_params.get('device'),
            dtype=llm_params.get('dtype', 'bfloat16'),
            max_new_tokens=llm_params.get('max_new_tokens', 256),
            temperature=llm_params.get('temperature', 0.7),
            top_p=llm_params.get('top_p', 0.9),
            repetition_penalty=llm_params.get('repetition_penalty', 1.05),
        )
        self._llm_client = get_client(self._llm_config)

        self.episode_logger = EpisodeLogger()
        
        # Event queue for processing
        self.event_queue = asyncio.Queue()
        
        # Monitoring data
        self.monitoring_data = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        self.openai_client = OpenAI() if OpenAI is not None else None
        self.router = get_router(openai_client=self.openai_client) if self.openai_client is not None else None

        # Simple simulated clock for predictive-mode demos (8am–8pm “waking day”)
        now = datetime.now()
        self.day_start_hour = 8
        self.day_end_hour = 20
        self.sim_time = now.replace(hour=self.day_start_hour, minute=0, second=0, microsecond=0)
        self.sim_seconds_per_step = 5 * 60  # 5 in-sim minutes per update tick
        self._wrapped_day = False

        # Cross-process monitor state (sim writes, Django reads)
        self.monitor_state_path = self._default_monitor_state_path()

        # Canonical locations for schedules (if spatial memory is absent/empty)
        self.default_locations = [
            "Hobbs Cafe",
            "The Willows Market and Pharmacy",
            "Harvey Oak Supply Store",
            "The Rose and Crown Pub",
            "Dorm for Oak Hill College",
            "Johnson Park",
        ]

        # Track commitment instances we've already evaluated (avoid repeated misses)
        self._commitment_instances_seen = set()
        
    def create_agent(self, name: str, folder_mem_saved: Optional[str] = None, profile_id: Optional[str] = None) -> Any:
        """
        Create a new agent (predictive or original)
        
        Args:
            name: Agent name
            folder_mem_saved: Path to saved memory folder
            profile_id: Optional persona profile id (e.g., 'enneagram_2')
            
        Returns:
            Agent instance
        """
        profile = get_profile(profile_id) if profile_id else None

        if self.use_predictive:
            agent = PredictivePersona(name, folder_mem_saved, profile=profile)
        else:
            # Lazy import: classic Persona pulls in OpenAI prompt stack.
            from persona.persona import Persona  # type: ignore
            agent = Persona(name, folder_mem_saved)
            # If we have a profile with scratch overrides, apply them for the classic Persona.
            # (PredictivePersona will still load scratch from disk if provided.)
            if profile and hasattr(agent, "scratch") and hasattr(agent.scratch, "__dict__"):
                scratch_over = profile.get("scratch_overrides") or {}
                if isinstance(scratch_over, dict):
                    for k, v in scratch_over.items():
                        if hasattr(agent.scratch, k):
                            setattr(agent.scratch, k, v)
        
        self.agents[name] = agent
        
        # Initialize monitoring data
        self.monitoring_data[name] = {
            'needs': {},
            'monologue': [],
            'predictions': [],
            'socialModels': []
        }
        
        return agent

    def start_new_session(self, session_id: str) -> None:
        """Rotate logging to a new session identifier."""
        self.episode_logger.start_session(session_id)
    
    def get_agent(self, name: str) -> Optional[Any]:
        """Get agent by name"""
        return self.agents.get(name)
    
    async def update_agents(self):
        """
        Main update loop for all agents
        Runs think cycles and processes actions
        """
        # Advance the shared simulation clock once per update tick
        self._advance_sim_time()

        tasks = []
        
        for name, agent in self.agents.items():
            if isinstance(agent, PredictivePersona):
                task = asyncio.create_task(self._update_predictive_agent(agent))
                tasks.append(task)
            else:
                # Handle original persona update
                task = asyncio.create_task(self._update_original_agent(agent))
                tasks.append(task)
        
        # Run all agent updates in parallel
        if tasks:
            await asyncio.gather(*tasks)
        
        # Send monitoring updates
        await self._send_monitoring_updates()

        # Persist any buffered episode data
        self.episode_logger.flush()
        self._persist_monitor_state()

    def _default_monitor_state_path(self) -> str:
        """
        Where the simulation writes the latest monitoring snapshot for the Django UI.
        """
        try:
            base = Path(__file__).resolve().parents[1]  # reverie/backend_server/persona -> backend_server
            out_dir = base / "distill_logs"
            out_dir.mkdir(parents=True, exist_ok=True)
            return str(out_dir / "monitor_state.json")
        except Exception:
            return "/tmp/social_ai_monitor_state.json"

    def _persist_monitor_state(self):
        """
        Persist monitoring state to a shared file so the Django process can read it.
        Atomic write: write to temp then rename.
        """
        try:
            path = self.monitor_state_path
            tmp = f"{path}.tmp"
            payload = {
                "timestamp": datetime.now().isoformat(),
                "sim_time": getattr(self, "sim_time", None).isoformat() if getattr(self, "sim_time", None) else None,
                "agents": self.monitoring_data,
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            # Don't let monitoring persistence crash the sim loop.
            pass

    def _advance_sim_time(self):
        """Advance simulation clock; wrap 8pm -> next day 8am."""
        self._wrapped_day = False
        self.sim_time = self.sim_time + timedelta(seconds=self.sim_seconds_per_step)
        if self.sim_time.hour >= self.day_end_hour:
            # Jump to next day start hour
            next_day = (self.sim_time + timedelta(days=1)).date()
            self.sim_time = datetime.combine(next_day, datetime.min.time()).replace(
                hour=self.day_start_hour, minute=0, second=0, microsecond=0
            )
            self._wrapped_day = True
    
    async def _update_predictive_agent(self, agent: PredictivePersona):
        """Update a single predictive agent"""
        try:
            # Sync agent time and enforce schedule if needed
            self._sync_agent_time(agent)
            await self._ensure_daily_schedule(agent)
            self._apply_schedule_location(agent)
            # Refresh nearby agents after location updates; needed for externalization + social actions.
            agent.nearby_agents = self._find_nearby_agents(agent)
            self._check_commitments(agent)

            # Run think cycle
            action = agent.think_cycle()

            # If heuristic returned no action, optionally ask teacher to choose via tool-call.
            # This primarily generates distillation data for the Qwen student.
            if not action:
                llm_action = await self._decide_action_with_llm(agent)
                if llm_action and llm_action.get("type") != "do_nothing":
                    action = llm_action
            
            # Process any actions
            if action:
                await self._process_agent_action(agent, action)
                if isinstance(agent, PredictivePersona):
                    metadata = agent.last_decision or {}
                    self.episode_logger.log_action(
                        agent_name=agent.name,
                        action=action,
                        needs=agent.needs.needs.copy(),
                        metadata=metadata,
                    )

            # Check for externalization
            external_thought = await self._check_externalization(agent)
            if external_thought:
                await self._process_external_speech(agent, external_thought)
            
            # Update monitoring data
            self._update_monitoring_data(agent)
            
        except Exception as e:
            print(f"Error updating agent {agent.name}: {e}")
            traceback.print_exc()

    def _sync_agent_time(self, agent: PredictivePersona):
        """Keep agent scratch time aligned to manager sim_time (helps prompts / UI)."""
        try:
            agent.scratch.curr_time = self.sim_time
        except Exception:
            pass

    def _get_allowed_locations(self, agent: PredictivePersona) -> List[str]:
        """Try to derive locations from spatial memory; fallback to defaults."""
        try:
            tree = getattr(agent, "s_mem", None)
            world = getattr(tree, "tree", {}) if tree else {}
            ville = world.get("the Ville", {})
            if isinstance(ville, dict) and ville:
                return sorted([k for k in ville.keys() if k])
        except Exception:
            pass
        return list(self.default_locations)

    async def _ensure_daily_schedule(self, agent: PredictivePersona):
        """
        Require a schedule if missing or stale (new day or wrapped day).
        A schedule covers the waking window only (default 12 hours).
        """
        today = self.sim_time.date()
        stale = (not getattr(agent, "daily_schedule", None)) or (getattr(agent, "schedule_for_date", None) != today)
        if stale:
            schedule = await self._create_schedule_with_llm(agent)
            if not schedule:
                schedule = self._fallback_schedule(agent)
            agent.daily_schedule = schedule
            agent.schedule_for_date = today
            agent._schedule_index = 0

    def _fallback_schedule(self, agent: PredictivePersona) -> List[Dict[str, Any]]:
        """Heuristic schedule when LLM is unavailable."""
        locs = self._get_allowed_locations(agent)
        day_minutes = (self.day_end_hour - self.day_start_hour) * 60
        work_loc = "Harvey Oak Supply Store" if "Harvey Oak Supply Store" in locs else (locs[0] if locs else "the Ville")
        home_loc = "Dorm for Oak Hill College" if "Dorm for Oak Hill College" in locs else (locs[-1] if locs else "the Ville")
        items = [
            {"activity": "morning routine", "location": home_loc, "duration_minutes": 60, "social_intent": ""},
            {"activity": "work shift", "location": work_loc, "duration_minutes": 360, "social_intent": "see coworkers / regulars"},
            {"activity": "lunch / break", "location": "Hobbs Cafe" if "Hobbs Cafe" in locs else work_loc, "duration_minutes": 60, "social_intent": "casual conversation"},
            {"activity": "finish work / errands", "location": work_loc, "duration_minutes": 180, "social_intent": ""},
            {"activity": "wind down", "location": home_loc, "duration_minutes": max(60, day_minutes - 660), "social_intent": ""},
        ]
        return self._normalize_schedule_items(items)

    def _normalize_schedule_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add start/end minutes and clamp duration totals to waking window."""
        day_minutes = (self.day_end_hour - self.day_start_hour) * 60
        out: List[Dict[str, Any]] = []
        t = 0
        for it in items:
            try:
                dur = int(it.get("duration_minutes", 0))
            except Exception:
                dur = 0
            if dur <= 0:
                continue
            if t >= day_minutes:
                break
            dur = min(dur, day_minutes - t)
            out.append(
                {
                    "activity": str(it.get("activity", ""))[:200],
                    "location": str(it.get("location", ""))[:200],
                    "duration_minutes": dur,
                    "start_minute": t,
                    "end_minute": t + dur,
                    "social_intent": str(it.get("social_intent", ""))[:200],
                }
            )
            t += dur
        return out

    async def _create_schedule_with_llm(self, agent: PredictivePersona) -> List[Dict[str, Any]]:
        """Call teacher with required schedule tool-call."""
        try:
            if self.router is None:
                return []
            tools = agent_schedule_tools()
            locs = self._get_allowed_locations(agent)
            day_minutes = (self.day_end_hour - self.day_start_hour) * 60

            prompt = (
                "Create a realistic schedule for the agent's NEXT waking block.\n"
                f"Waking window: {self.day_start_hour}:00 to {self.day_end_hour}:00 ({day_minutes} minutes).\n"
                "The agent maintains a schedule and may also have longer-horizon commitments; incorporate them.\n"
                "Return the schedule by calling the provided tool.\n\n"
                f"Agent: {agent.name}\n"
                f"Persona style:\n{profile_style_text(getattr(agent, 'profile', None))}\n\n"
                f"Need states: {agent.needs.needs}\n"
                f"Bio (learned): {getattr(agent.scratch, 'learned', '')}\n"
                f"Currently: {getattr(agent.scratch, 'currently', '')}\n"
                f"Daily plan requirement: {getattr(agent.scratch, 'daily_plan_req', '')}\n\n"
                f"Upcoming commitments (recurring/one-off): {self._summarize_commitments(agent, horizon_days=14)}\n\n"
                f"Allowed locations (choose exact strings): {locs}\n\n"
                "Guidance:\n"
                "- Include at least one work/professional block.\n"
                "- Include at least one social opportunity if connection is low.\n"
                "- Durations should be in minutes; total should be close to the full window.\n"
            )

            messages = [
                {"role": "system", "content": "Choose a daily schedule by calling the provided tool."},
                {"role": "user", "content": prompt},
            ]

            msg, _debug = await asyncio.to_thread(
                self.router.chat,
                provider=self.router.provider_for_task("schedule"),
                task="schedule",
                messages=messages,
                temperature=0.4,
                max_tokens=600,
                tools=tools,
                tool_choice="required",
                meta={"agent": agent.name},
            )

            tc = extract_tool_call(msg)
            if not tc:
                return []
            tool_name, args = tc
            call = tool_call_to_schedule(tool_name, args)
            raw_items = call.get("items") or []
            if not isinstance(raw_items, list):
                return []

            # Validate + normalize; enforce allowed locations
            allowed = set(locs)
            cleaned = []
            for it in raw_items:
                if not isinstance(it, dict):
                    continue
                loc = it.get("location")
                if loc not in allowed:
                    continue
                cleaned.append(it)
            return self._normalize_schedule_items(cleaned)
        except Exception as e:
            print(f"LLM schedule error for {agent.name}: {e}")
            return []

    def _summarize_commitments(self, agent: PredictivePersona, horizon_days: int = 7) -> List[Dict[str, Any]]:
        """
        Return a compact, JSON-serializable summary of upcoming commitments.
        Commitment schema (best-effort):
          {"id","title","location","participants","start_iso","end_iso","recurrence": {"freq":"daily|weekly","days_of_week":[0..6]}}
        """
        try:
            commits = getattr(agent, "commitments", None) or []
        except Exception:
            commits = []
        if not isinstance(commits, list):
            return []
        out = []
        now = self.sim_time
        end = now + timedelta(days=horizon_days)
        for c in commits[:50]:
            if not isinstance(c, dict):
                continue
            title = c.get("title") or c.get("name") or "commitment"
            loc = c.get("location")
            rec = c.get("recurrence")
            participants = c.get("participants") or []
            start_iso = c.get("start_iso") or c.get("start") or None
            # Keep it compact; schedule planner just needs “what exists”
            out.append(
                {
                    "id": c.get("id") or title,
                    "title": title,
                    "location": loc,
                    "participants": participants,
                    "start_iso": start_iso,
                    "recurrence": rec,
                }
            )
        return out

    def _check_commitments(self, agent: PredictivePersona):
        """
        If a commitment is due now-ish and agent isn't at the expected location, emit missed_commitment.
        This is a minimal expectation mechanism that can later be tied into richer movement/execution.
        """
        commits = getattr(agent, "commitments", None) or []
        if not isinstance(commits, list) or not commits:
            return

        now = self.sim_time
        for c in commits:
            if not isinstance(c, dict):
                continue
            loc = c.get("location")
            if not loc:
                continue

            # Determine if this commitment applies today at the current hour/minute.
            due = False
            # One-off: start_iso
            start_iso = c.get("start_iso") or c.get("start")
            if start_iso:
                try:
                    start_dt = datetime.fromisoformat(str(start_iso))
                    # Treat due within the current tick window
                    if abs((start_dt - now).total_seconds()) <= self.sim_seconds_per_step:
                        due = True
                        inst_key = f"{agent.name}:{c.get('id') or c.get('title') or loc}:{start_dt.isoformat()}"
                except Exception:
                    inst_key = None
            else:
                # Recurring: daily/weekly with "time_hm"
                rec = c.get("recurrence") or {}
                time_hm = c.get("time_hm") or rec.get("time_hm")
                freq = rec.get("freq")
                inst_key = None
                if time_hm and freq:
                    try:
                        hh, mm = [int(x) for x in str(time_hm).split(":", 1)]
                        if now.hour == hh and now.minute == mm:
                            if freq == "daily":
                                due = True
                            elif freq == "weekly":
                                dow = now.weekday()
                                days = rec.get("days_of_week") or []
                                if dow in days:
                                    due = True
                            inst_key = f"{agent.name}:{c.get('id') or c.get('title') or loc}:{now.date().isoformat()}:{hh:02d}:{mm:02d}"
                    except Exception:
                        pass

            if not due:
                continue
            if inst_key and inst_key in self._commitment_instances_seen:
                continue
            if inst_key:
                self._commitment_instances_seen.add(inst_key)

            # Attendance check
            if agent.current_location != loc:
                participants = c.get("participants") or []
                title = c.get("title") or "commitment"
                event = {
                    "type": "missed_commitment",
                    "who": agent.name,
                    "title": title,
                    "expected_location": loc,
                    "actual_location": agent.current_location,
                    "participants": participants,
                    "timestamp": now,
                }
                # Agent experiences the miss
                agent.observe_event(event)
                # Others who were expecting can observe too
                for pname in participants:
                    other = self.agents.get(pname)
                    if isinstance(other, PredictivePersona):
                        other.observe_event(event)

    def _apply_schedule_location(self, agent: PredictivePersona):
        """Set agent.current_location based on schedule and current sim_time."""
        sched = getattr(agent, "daily_schedule", None) or []
        if not sched:
            return
        minutes = (self.sim_time.hour - self.day_start_hour) * 60 + self.sim_time.minute
        # Find current item
        chosen = None
        for it in sched:
            if it.get("start_minute", 0) <= minutes < it.get("end_minute", 0):
                chosen = it
                break
        if chosen is None:
            return
        new_loc = chosen.get("location")
        if new_loc and new_loc != agent.current_location:
            agent.current_location = new_loc
            # Record a lightweight event for later learning/monitoring
            agent.recent_events.append({"type": "arrive_location", "location": new_loc, "timestamp": self.sim_time})
    
    async def _update_original_agent(self, agent: Any):
        """Update original persona (compatibility mode)"""
        # This would integrate with the existing persona update logic
        pass
    
    async def _process_agent_action(self, agent: PredictivePersona, action: Dict[str, Any]):
        """Process an action decided by an agent"""
        action_type = action.get('type')
        
        if action_type == 'initiate_conversation':
            # Find nearby agents
            nearby = self._find_nearby_agents(agent)
            if nearby:
                target_name = action.get("target_name")
                target = None
                if target_name:
                    for a in nearby:
                        if a.name == target_name:
                            target = a
                            break
                if target is None:
                    target = nearby[0]  # Pick first available
                await self._initiate_conversation(agent, target)
                # Post-interaction schedule decision for both parties
                await self._post_interaction_schedule_decision(
                    agent, interaction_summary=f"Started a conversation with {target.name}"
                )
                await self._post_interaction_schedule_decision(
                    target, interaction_summary=f"Had a conversation started by {agent.name}"
                )
                
        elif action_type == 'ask_for_help':
            await self._broadcast_help_request(agent)
            await self._post_interaction_schedule_decision(
                agent, interaction_summary="Asked nearby people for help"
            )
            
        elif action_type == 'offer_help':
            # Find agents who might need help
            needy_agents = self._find_agents_needing_help(agent)
            if needy_agents:
                target_name = action.get("target_name")
                target = None
                if target_name:
                    for a in needy_agents:
                        if a.name == target_name:
                            target = a
                            break
                if target is None:
                    target = needy_agents[0]
                await self._offer_help_to(agent, target)
                await self._post_interaction_schedule_decision(
                    agent, interaction_summary=f"Offered help to {target.name}"
                )
                await self._post_interaction_schedule_decision(
                    target, interaction_summary=f"Received an offer of help from {agent.name}"
                )
                
        # Add more action handlers as needed
    
    async def _check_externalization(self, agent: PredictivePersona) -> Optional[str]:
        """
        Check if agent should externalize thoughts and generate speech
        """
        # Check conditions for externalization
        if (agent.needs.needs['connection'] < self.config.get_externalization_threshold() 
            and agent.nearby_agents):
            
            # Generate prompt for LLM
            prompt = create_need_aware_prompt(
                'externalize',
                agent_name=agent.name,
                internal_monologue=agent.monologue.get_recent_thoughts(10),
                need_states=agent.needs.needs,
                predictions=list(agent.prediction_buffer)[-5:],
                persona_style=profile_style_text(getattr(agent, "profile", None)),
                context={
                    'location': agent.current_location,
                    'nearby_agents': [a.name for a in agent.nearby_agents],
                    'recent_events': [e.get('type', 'unknown') for e in agent.recent_events][-5:]
                }
            )
            
            # Call LLM
            try:
                response = await self._call_llm(prompt)
                
                # Validate response
                is_valid, cleaned, issues = validate_llm_response(
                    response, 
                    agent.needs.needs,
                    {'nearby_agents': agent.nearby_agents}
                )
                
                if is_valid:
                    return cleaned
                else:
                    print(f"Invalid response for {agent.name}: {issues}")
                    
            except Exception as e:
                print(f"LLM error for {agent.name}: {e}")
        
        return None
    
    async def _call_llm(self, prompt: str) -> str:
        """Call teacher model through ModelRouter, falling back to local client."""
        try:
            if self.router is not None:
                messages = [
                    {"role": "system", "content": "You are a realistic human-like agent."},
                    {"role": "user", "content": prompt},
                ]

                msg, _debug = await asyncio.to_thread(
                    self.router.chat,
                    provider=self.router.provider_for_task("externalize"),
                    task="externalize",
                    messages=messages,
                    temperature=0.8,
                    max_tokens=150,
                )
                return (msg.get("content") or "").strip()

            # Fallback: local LLM client
            response_text = await asyncio.to_thread(
                self._llm_client.generate,
                [
                    {"role": "system", "content": "You are a realistic human-like agent."},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=150,
                temperature=0.8,
            )
            return response_text.strip()

        except Exception as e:
            print(f"LLM error: {e}")
            return "..."

    async def _decide_action_with_llm(self, agent: PredictivePersona) -> Optional[Dict[str, Any]]:
        """
        Optional: Use teacher to pick an external action via tool-calls.
        This is primarily for distillation data collection (tool calls + dialogue shaping).
        """
        try:
            if self.router is None:
                return None
            nearby_names = [a.name for a in self._find_nearby_agents(agent)]
            tools = agent_action_tools()

            prompt = (
                "You are selecting ONE external action for the agent.\n"
                "If no good action exists, call do_nothing.\n\n"
                f"Agent: {agent.name}\n"
                f"Location: {agent.current_location}\n"
                f"Nearby agents: {nearby_names}\n"
                f"Need states: {agent.needs.needs}\n"
                f"Recent events: {[e.get('type','unknown') for e in list(agent.recent_events)[-5:]]}\n"
                f"Recent thoughts: {agent.monologue.get_recent_thoughts(5)}\n"
                f"Upcoming commitments: {self._summarize_commitments(agent, horizon_days=14)}\n"
            )

            messages = [
                {"role": "system", "content": "Choose actions by calling the provided tools."},
                {"role": "user", "content": prompt},
            ]

            msg, _debug = await asyncio.to_thread(
                self.router.chat,
                provider=self.router.provider_for_task("action"),
                task="action",
                messages=messages,
                temperature=0.2,
                max_tokens=200,
                tools=tools,
                tool_choice="auto",
                meta={"agent": agent.name},
            )

            tc = extract_tool_call(msg)
            if not tc:
                return None
            tool_name, args = tc
            action = tool_call_to_action(tool_name, args)

            # Basic guardrails: if tool references unknown target, do nothing.
            if action.get("type") in ("initiate_conversation", "offer_help"):
                if action.get("target_name") and action["target_name"] not in nearby_names:
                    return {"type": "do_nothing"}
            return action
        except Exception as e:
            print(f"LLM action selection error for {agent.name}: {e}")
            return None
    
    async def _process_external_speech(self, agent: PredictivePersona, speech: str):
        """Process externalized speech from an agent"""
        # Create speech event
        event = {
            'type': 'agent_speech',
            'speaker': agent.name,
            'content': speech,
            'location': agent.current_location,
            'timestamp': datetime.now()
        }

        # Record on the speaking agent for monitoring/debug (without treating as a need-impacting event)
        try:
            if hasattr(agent, "recent_speech"):
                agent.recent_speech.append(
                    {
                        "speaker": agent.name,
                        "content": speech,
                        "location": agent.current_location,
                        "timestamp": event["timestamp"],
                    }
                )
        except Exception:
            pass
        
        # Notify nearby agents
        for other_agent in agent.nearby_agents:
            if isinstance(other_agent, PredictivePersona):
                other_agent.observe_event(event)
        
        # Add to event queue for environment processing
        await self.event_queue.put(event)

        thoughts = agent.monologue.get_recent_thoughts(5)
        nearby_names = [other.name for other in agent.nearby_agents if isinstance(other, PredictivePersona)]
        self.episode_logger.log_speech(
            agent_name=agent.name,
            content=speech,
            needs=agent.needs.needs.copy(),
            location=agent.current_location,
            monologue=thoughts,
            nearby_agents=nearby_names,
        )
    
    def _find_nearby_agents(self, agent: PredictivePersona) -> List[PredictivePersona]:
        """Find agents near the given agent"""
        nearby = []
        for name, other in self.agents.items():
            if name != agent.name and isinstance(other, PredictivePersona):
                # Check if in same location or adjacent
                if self._are_agents_nearby(agent, other):
                    nearby.append(other)
        return nearby
    
    def _are_agents_nearby(self, agent1: PredictivePersona, agent2: PredictivePersona) -> bool:
        """Check if two agents are nearby"""
        # This would integrate with the spatial system
        # For now, simple check
        return agent1.current_location == agent2.current_location
    
    def _find_agents_needing_help(self, helper: PredictivePersona) -> List[PredictivePersona]:
        """Find agents who might need help based on their need states"""
        needy = []
        
        for name, agent in self.agents.items():
            if name != helper.name and isinstance(agent, PredictivePersona):
                # Check if any critical needs
                critical_needs = [
                    need for need, value in agent.needs.needs.items()
                    if value < self.config.get_critical_thresholds()[need]
                ]
                
                if critical_needs:
                    needy.append(agent)
        
        return needy
    
    async def _initiate_conversation(self, initiator: PredictivePersona, 
                                   target: PredictivePersona):
        """Handle conversation initiation between agents"""
        # Create interaction event
        event = {
            'type': 'conversation_starts',
            'initiator': initiator.name,
            'target': target.name,
            'location': initiator.current_location,
            'timestamp': datetime.now()
        }
        
        # Both agents observe the event
        initiator.observe_event(event)
        target.observe_event(event)
        
        # Generate responses using social interaction prompts
        for agent, other in [(initiator, target), (target, initiator)]:
            response = agent.interact_with_agent(other, 'greeting')
            
            if response.get('content'):
                await self._process_external_speech(agent, response['content'])
    
    async def _broadcast_help_request(self, agent: PredictivePersona):
        """Broadcast a help request from an agent"""
        event = {
            'type': 'help_request',
            'requester': agent.name,
            'location': agent.current_location,
            'timestamp': datetime.now()
        }
        
        # Notify all nearby agents
        for other in self._find_nearby_agents(agent):
            other.observe_event(event)
    
    async def _offer_help_to(self, helper: PredictivePersona, 
                           target: PredictivePersona):
        """Handle help offering between agents"""
        event = {
            'type': 'help_offered',
            'helper': helper.name,
            'target': target.name,
            'timestamp': datetime.now()
        }
        
        helper.observe_event(event)
        target.observe_event(event)

    async def _post_interaction_schedule_decision(self, agent: PredictivePersona, interaction_summary: str):
        """
        After an interaction, force the agent to either keep their schedule or edit it.
        If no LLM is configured, default to sticking with the schedule.
        """
        try:
            if self.router is None:
                return
            # If agent has no schedule yet, let the normal schedule enforcement handle it.
            if not getattr(agent, "daily_schedule", None):
                return

            # Step 0: optionally create a commitment (then schedule decision can react to it)
            await self._post_interaction_commitment_update(agent, interaction_summary)

            # Current minute in waking window
            minutes_now = (self.sim_time.hour - self.day_start_hour) * 60 + self.sim_time.minute
            day_minutes = (self.day_end_hour - self.day_start_hour) * 60
            remaining = max(0, day_minutes - max(0, minutes_now))
            if remaining < 30:
                # Too little time to meaningfully re-plan.
                return

            locs = self._get_allowed_locations(agent)
            tools = agent_schedule_adjust_tools()

            # Provide current schedule context (truncate for prompt size)
            sched = list(getattr(agent, "daily_schedule", []))
            sched_preview = [
                {k: it.get(k) for k in ("activity", "location", "duration_minutes", "start_minute", "end_minute")}
                for it in sched[:12]
            ]

            prompt = (
                "An interaction just completed. Decide whether to keep the schedule unchanged, "
                "or to edit the remaining schedule for today.\n\n"
                f"Agent: {agent.name}\n"
                f"Time: {self.sim_time.isoformat()}\n"
                f"Current location: {agent.current_location}\n"
                f"Interaction summary: {interaction_summary}\n"
                f"Need states: {agent.needs.needs}\n\n"
                f"Upcoming commitments: {self._summarize_commitments(agent, horizon_days=14)}\n\n"
                f"Minutes remaining in waking window: {remaining}\n"
                f"Allowed locations (choose exact strings): {locs}\n\n"
                f"Current schedule (preview): {sched_preview}\n\n"
                "Call exactly one tool:\n"
                "- stick_with_schedule\n"
                "- edit_schedule (provide new items for the remaining time only)\n"
            )

            messages = [
                {"role": "system", "content": "Choose by calling one tool."},
                {"role": "user", "content": prompt},
            ]

            msg, _debug = await asyncio.to_thread(
                self.router.chat,
                provider="teacher",
                task="schedule",
                messages=messages,
                temperature=0.3,
                max_tokens=450,
                tools=tools,
                tool_choice="required",
                meta={"agent": agent.name, "phase": "post_interaction"},
            )

            tc = extract_tool_call(msg)
            if not tc:
                return
            tool_name, args = tc
            decision = tool_call_to_schedule_adjust(tool_name, args)
            if decision.get("type") == "stick_with_schedule":
                return

            # Edit schedule: rewrite remaining items from now until end of day
            raw_items = decision.get("items") or []
            if not isinstance(raw_items, list):
                return

            allowed = set(locs)
            cleaned = []
            for it in raw_items:
                if not isinstance(it, dict):
                    continue
                loc = it.get("location")
                if loc not in allowed:
                    continue
                cleaned.append(it)

            remainder = self._normalize_schedule_items(cleaned)
            if not remainder:
                return

            # Splice: keep any already-started portion, then replace from "now"
            prefix = []
            for it in sched:
                if it.get("end_minute", 0) <= minutes_now:
                    prefix.append(it)
                else:
                    break

            # Re-base remainder start/end minutes from minutes_now
            rebased = []
            t = minutes_now
            for it in remainder:
                dur = int(it.get("duration_minutes", 0))
                if dur <= 0 or t >= day_minutes:
                    break
                dur = min(dur, day_minutes - t)
                rebased.append(
                    {
                        **it,
                        "start_minute": t,
                        "end_minute": t + dur,
                        "duration_minutes": dur,
                    }
                )
                t += dur

            agent.daily_schedule = prefix + rebased
        except Exception as e:
            print(f"Post-interaction schedule decision error for {agent.name}: {e}")

    async def _post_interaction_commitment_update(self, agent: PredictivePersona, interaction_summary: str):
        """
        After an interaction, allow the agent to add a future commitment (or explicitly choose none).
        This runs before the schedule stick/edit choice so the schedule decision can incorporate it.
        """
        if self.router is None:
            return
        try:
            locs = self._get_allowed_locations(agent)
            agent_names = list(self.agents.keys())
            tools = agent_commitment_tools()

            prompt = (
                "An interaction just completed. Decide if you should create a NEW future commitment.\n"
                "If not, call no_new_commitment.\n\n"
                f"Agent: {agent.name}\n"
                f"Time: {self.sim_time.isoformat()}\n"
                f"Location: {agent.current_location}\n"
                f"Interaction summary: {interaction_summary}\n"
                f"Need states: {agent.needs.needs}\n\n"
                f"Existing commitments: {self._summarize_commitments(agent, horizon_days=30)}\n\n"
                f"Allowed locations (choose exact strings): {locs}\n"
                f"Known agents (for participants): {agent_names}\n\n"
                "Only create commitments that are plausible and that you intend to keep.\n"
                "Commitments can be one-off (specific date/time) or recurring (e.g., daily work block).\n"
                "If a commitment involves another agent, include them as a participant (they may expect you).\n"
            )

            messages = [
                {"role": "system", "content": "Choose by calling one tool."},
                {"role": "user", "content": prompt},
            ]

            msg, _debug = await asyncio.to_thread(
                self.router.chat,
                provider="teacher",
                task="schedule",
                messages=messages,
                temperature=0.25,
                max_tokens=350,
                tools=tools,
                tool_choice="required",
                meta={"agent": agent.name, "phase": "post_interaction_commitment"},
            )

            tc = extract_tool_call(msg)
            if not tc:
                return
            tool_name, args = tc
            action = tool_call_to_commitment_action(tool_name, args)
            if action.get("type") != "add_commitment":
                return

            c = action.get("commitment") or {}
            if not isinstance(c, dict):
                return
            title = (c.get("title") or "").strip()
            loc = c.get("location")
            if not title or loc not in set(locs):
                return

            # Validate participants (must be existing agents; exclude self)
            participants = c.get("participants") or []
            if not isinstance(participants, list):
                participants = []
            participants = [p for p in participants if p in self.agents and p != agent.name]

            # Normalize id
            cid = (c.get("id") or f"c_{agent.name.lower().replace(' ','_')}_{int(self.sim_time.timestamp())}").strip()

            new_c = {
                "id": cid,
                "title": title[:120],
                "location": loc,
                "participants": participants,
            }
            # Pass through time fields if present (best-effort)
            if c.get("start_iso"):
                new_c["start_iso"] = str(c.get("start_iso"))
            if c.get("end_iso"):
                new_c["end_iso"] = str(c.get("end_iso"))
            if isinstance(c.get("recurrence"), dict):
                new_c["recurrence"] = c.get("recurrence")

            # Append if not duplicate id
            commits = getattr(agent, "commitments", None)
            if not isinstance(commits, list):
                commits = []
            if any(isinstance(x, dict) and x.get("id") == cid for x in commits):
                return
            commits.append(new_c)
            agent.commitments = commits
        except Exception as e:
            print(f"Post-interaction commitment update error for {agent.name}: {e}")
    
    def _update_monitoring_data(self, agent: PredictivePersona):
        """Update monitoring data for visualization"""
        self.monitoring_data[agent.name] = {
            'needs': agent.needs.needs.copy(),
            'monologue': [
                {'type': t['type'], 'content': t['content']}
                for t in list(agent.monologue.thoughts)[-10:]
            ],
            'speech': list(getattr(agent, "recent_speech", []))[-10:],
            'predictions': [
                {
                    'event': pred.get('event', 'unknown'),
                    'probability': pred.get('probability', 0),
                    'impacts': self._format_impacts(pred.get('need_impacts', {}))
                }
                for pred in list(agent.prediction_buffer)[-5:]
            ],
            'socialModels': [
                f"{other_name}: {self._summarize_needs(model['estimated_needs'])}"
                for other_name, model in list(agent.other_agent_models.items())[:5]
            ],
            'time': self.sim_time.isoformat(),
            'location': agent.current_location,
            'schedule': list(getattr(agent, "daily_schedule", [])[:10]),
        }
    
    def _format_impacts(self, impacts: Dict[str, float]) -> str:
        """Format need impacts for display"""
        parts = []
        for need, change in sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
            if abs(change) > 0.05:
                direction = "↑" if change > 0 else "↓"
                parts.append(f"{need}{direction}")
        return ", ".join(parts)
    
    def _summarize_needs(self, needs: Dict[str, float]) -> str:
        """Summarize need states for display"""
        # Find most deficient need
        deficient = min(needs.items(), key=lambda x: x[1])
        if deficient[1] < 0.3:
            return f"needs {deficient[0]} (critical)"
        elif deficient[1] < 0.5:
            return f"needs {deficient[0]} (low)"
        else:
            return "appears satisfied"
    
    async def _send_monitoring_updates(self):
        """Send monitoring updates to connected websockets"""
        if self.websocket_connections:
            update_data = {
                'timestamp': datetime.now().isoformat(),
                'agents': self.monitoring_data
            }
            
            # Send to all connected clients
            disconnected = set()
            for ws in self.websocket_connections:
                try:
                    await ws.send(json.dumps(update_data))
                except:
                    disconnected.add(ws)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected
    
    def add_websocket(self, ws):
        """Add a websocket connection for monitoring"""
        self.websocket_connections.add(ws)
    
    def remove_websocket(self, ws):
        """Remove a websocket connection"""
        self.websocket_connections.discard(ws)
    
    async def save_all_agents(self, base_path: str):
        """Save all agent states"""
        for name, agent in self.agents.items():
            if isinstance(agent, PredictivePersona):
                filepath = f"{base_path}/{name}_state.json"
                agent.save_state(filepath)
    
    def get_agent_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get a summary of an agent's current state"""
        agent = self.agents.get(agent_name)
        if not agent or not isinstance(agent, PredictivePersona):
            return {}
        
        return {
            'name': agent.name,
            'needs': agent.needs.needs,
            'recent_thoughts': agent.monologue.get_recent_thoughts(5),
            'current_predictions': [
                {
                    'event': p.get('event'),
                    'probability': p.get('probability')
                }
                for p in list(agent.prediction_buffer)[-3:]
            ],
            'location': agent.current_location,
            'nearby_agents': [a.name for a in agent.nearby_agents]
        }


# Global manager instance
_manager_instance = None

def get_manager(use_predictive: bool = True) -> PersonaManager:
    """Get global persona manager instance"""
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = PersonaManager(use_predictive)
    
    return _manager_instance


async def run_simulation_step():
    """Run one step of the simulation"""
    manager = get_manager()
    await manager.update_agents()

"""
Judge Agent -- evaluates the need-impact of agent actions via GPT-5.2.

Two components:
- JudgeCache: semantic similarity cache backed by embeddings (avoids
  re-evaluating similar activities).
- JudgeAgent: builds the evaluation prompt, calls the model, parses the
  structured verdict.

The judge is called ~every 5 sim-minutes per agent. Verdicts are cached so
that "working on project planning at Hobbs Cafe" and "working on project
updates at Hobbs Cafe" share the same verdict (cosine similarity > threshold).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Semantic verdict cache
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class JudgeCache:
    """
    LRU cache of judge verdicts keyed by activity description embeddings.

    On lookup, the cache computes cosine similarity between the query
    embedding and all cached entries.  If any entry exceeds the similarity
    threshold, its verdict is returned (cache hit).
    """

    def __init__(self, max_size: int = 500, similarity_threshold: float = 0.85,
                 persist_path: Optional[str] = None):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.persist_path = persist_path

        # OrderedDict for LRU: key = hash(desc), value = (embedding, verdict)
        self._cache: OrderedDict[str, Tuple[List[float], Dict[str, Any]]] = OrderedDict()
        self._hits = 0
        self._misses = 0

        # Load persisted cache if available
        if persist_path and os.path.exists(persist_path):
            try:
                with open(persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data.get("entries", []):
                    key = entry["key"]
                    emb = entry["embedding"]
                    verdict = entry["verdict"]
                    self._cache[key] = (emb, verdict)
            except Exception:
                pass

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}

    def lookup(self, embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        Find a cached verdict whose embedding is similar enough to the query.
        Returns the verdict dict on hit, None on miss.
        """
        best_sim = 0.0
        best_key: Optional[str] = None

        for key, (cached_emb, _verdict) in self._cache.items():
            sim = _cosine_similarity(embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_sim >= self.similarity_threshold and best_key is not None:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(best_key)
            return self._cache[best_key][1]

        self._misses += 1
        return None

    def store(self, description: str, embedding: List[float], verdict: Dict[str, Any]) -> None:
        """Store a verdict in the cache and persist to disk."""
        key = hashlib.sha256(description.encode("utf-8")).hexdigest()[:16]
        self._cache[key] = (embedding, verdict)
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

        # Auto-persist after every new entry
        self.save()

    def save(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            entries = []
            for key, (emb, verdict) in self._cache.items():
                entries.append({"key": key, "embedding": emb, "verdict": verdict})
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump({"entries": entries, "stats": self.stats}, f)
        except Exception as e:
            import sys
            print(f"[judge] Cache save failed ({self.persist_path}): {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Judge agent
# ---------------------------------------------------------------------------

# The nine need names (must match need_config.yaml).
NEED_NAMES = [
    "connection", "safety", "approval", "empathy",
    "fun", "attention", "achievement", "autonomy", "purpose",
]

# Magnitude -> per-step scale factor applied to raw impact values.
MAGNITUDE_SCALES = {
    "negligible": 0.0005,
    "small": 0.002,
    "moderate": 0.004,
    "significant": 0.007,
    "major": 0.01,
}

_SYSTEM_PROMPT = """\
You are a simulation judge evaluating how an agent's current activity affects \
their psychological needs.  You produce ONLY a JSON object -- no other text.

The agent lives in a small-town simulation.  Consider:
- Positive impacts: the activity naturally fulfills certain needs
- Negative impacts: mistakes, social friction, boredom from repetition, \
  fatigue, environmental mishaps, being ignored or rejected
- Neutral: some needs are simply unaffected

Be realistic and varied.  Not every action is positive.  Repeated identical \
activities should show diminishing returns.  Social interactions can go wrong. \
Solo work can be lonely.

## Calibration scale

Use these reference points to anchor your impact values consistently:

| Activity example                  | Key impacts (non-zero)                                     | Magnitude   |
|-----------------------------------|------------------------------------------------------------|-------------|
| sleeping quietly                  | all ~0.0                                                   | negligible  |
| morning routine (shower, dress)   | safety +0.1, autonomy +0.1                                 | small       |
| casual conversation with a friend | connection +0.5, fun +0.3, attention +0.2                   | moderate    |
| completing an important work task | achievement +0.7, purpose +0.5, approval +0.2               | significant |
| being publicly criticized         | approval -0.8, safety -0.4, connection -0.3                 | significant |
| having a meal alone               | safety +0.2, fun +0.1, connection -0.1                      | small       |
| helping someone in need           | empathy +0.6, purpose +0.5, connection +0.4                 | moderate    |
| being ignored in a group          | attention -0.6, approval -0.4, connection -0.3              | moderate    |

All other activities should fall within this scale.  A +0.5 always means \
the same moderate positive effect regardless of which call this is.

## Output schema

Return JSON with this exact schema:
{
  "impacts": {<need_name>: <float -1.0 to 1.0>, ...},
  "narrative": "<1-2 sentence first-person subjective account of how this went>",
  "magnitude": "<negligible|small|moderate|significant|major>"
}

Rules:
- Include ALL 9 needs in "impacts": connection, safety, approval, empathy, \
  fun, attention, achievement, autonomy, purpose
- Values range from -1.0 (strong negative) to +1.0 (strong positive), 0.0 = no effect
- "narrative" should be written as the agent's inner voice (first person, subjective)
- "magnitude" reflects overall significance of this moment
- Return ONLY the JSON object, no markdown fences or explanation
"""


def _build_user_prompt(
    persona_name: str,
    activity: str,
    location: str,
    nearby_agents: List[str],
    current_needs: Dict[str, float],
    profile_label: str,
    core_motive: str,
    recent_monologue: List[str],
    chatting_with: Optional[str],
    ledger: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the user-turn prompt describing the situation to evaluate."""
    needs_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(current_needs.items()))
    nearby_str = ", ".join(nearby_agents) if nearby_agents else "no one"
    mono_str = " | ".join(recent_monologue[-3:]) if recent_monologue else "(none)"
    chat_str = f"Currently in conversation with {chatting_with}.\n" if chatting_with else ""

    # Format ledger entries as prior-verdict anchors
    ledger_str = ""
    if ledger:
        lines = []
        for entry in ledger[-5:]:  # last 5 for context
            act = entry.get("activity", "?")[:60]
            mag = entry.get("magnitude", "?")
            impacts = entry.get("impacts", {})
            # Show only non-zero impacts, sorted by absolute value
            nonzero = [(k, v) for k, v in impacts.items() if abs(v) >= 0.05]
            nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
            impact_str = ", ".join(f"{k}={v:+.1f}" for k, v in nonzero[:4])
            lines.append(f'- "{act}" -> {impact_str} [{mag}]')
        if lines:
            ledger_str = (
                "\nYour recent evaluations for this agent (maintain consistent scale):\n"
                + "\n".join(lines) + "\n"
            )

    return f"""\
Agent: {persona_name}
Personality: {profile_label} -- {core_motive}
Current activity: {activity}
Location: {location}
Nearby agents: {nearby_str}
{chat_str}Current need levels (0=empty, 1=full): {needs_str}
Recent inner thoughts: {mono_str}
{ledger_str}
Evaluate how this activity is going for {persona_name} right now. \
Consider both what could go well and what could go wrong."""


class JudgeAgent:
    """
    Calls GPT-5.2 (via ModelRouter) to evaluate an agent's action and
    produce a structured need-impact verdict.
    """

    def __init__(self, cache: Optional[JudgeCache] = None):
        self._cache = cache or JudgeCache()
        self._router = None
        self._openai_client = None

    def _get_router(self):
        """Lazy-init the model router."""
        if self._router is not None:
            return self._router
        try:
            from models.router import get_router  # type: ignore
            from openai import OpenAI  # type: ignore
            self._openai_client = OpenAI()
            self._router = get_router(openai_client=self._openai_client)
            return self._router
        except Exception as e:
            # Log the failure once so we know why the judge can't call the API
            if not getattr(self, "_router_error_logged", False):
                import sys
                print(f"[judge] ModelRouter init failed: {e}", file=sys.stderr)
                self._router_error_logged = True
            return None

    def _get_embedding(self, text: str) -> List[float]:
        """Get an embedding for cache lookup."""
        try:
            from persona.prompt_template.gpt_structure import get_embedding
            return get_embedding(text)
        except Exception:
            # Deterministic fallback
            import hashlib as _h, random as _r
            seed = int(_h.sha256(text.encode()).hexdigest()[:16], 16)
            rng = _r.Random(seed)
            return [rng.uniform(-1.0, 1.0) for _ in range(1536)]

    def evaluate(
        self,
        persona_name: str,
        activity: str,
        location: str,
        nearby_agents: List[str],
        current_needs: Dict[str, float],
        profile_label: str = "",
        core_motive: str = "",
        recent_monologue: List[str] = None,
        chatting_with: Optional[str] = None,
        ledger: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate an action and return a verdict dict:
          {"impacts": {...}, "narrative": "...", "magnitude": "...", "cached": bool}

        The ``ledger`` parameter is a list of this agent's prior verdicts,
        used as few-shot calibration anchors so the model maintains a
        consistent numeric scale across calls.

        Returns None if evaluation fails (API error, etc.).
        """
        if not activity or "sleep" in activity.lower():
            return None

        # Build a cache key from activity + location (agent-independent so
        # "Nina working at cafe" and "Jasper working at cafe" share a cache
        # entry for the base impact, with the narrative being generic).
        cache_desc = f"{activity} @ {location}"
        embedding = self._get_embedding(cache_desc)

        # Cache lookup
        cached = self._cache.lookup(embedding)
        if cached is not None:
            return {**cached, "cached": True}

        # Cache miss -- call the judge model
        router = self._get_router()
        if router is None:
            return None

        user_prompt = _build_user_prompt(
            persona_name=persona_name,
            activity=activity,
            location=location,
            nearby_agents=nearby_agents,
            current_needs=current_needs,
            profile_label=profile_label,
            core_motive=core_motive,
            recent_monologue=recent_monologue or [],
            chatting_with=chatting_with,
            ledger=ledger,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            msg, _debug = router.chat(
                provider=router.provider_for_task("judge"),
                task="judge",
                messages=messages,
                temperature=0.7,
                max_tokens=350,
                meta={"agent": persona_name, "activity": activity[:80], **(meta or {})},
            )

            # Parse the response
            content = msg.get("content", "") or ""
            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            verdict = json.loads(content)

            # Validate and normalize
            impacts = verdict.get("impacts", {})
            if not isinstance(impacts, dict):
                return None

            # Ensure all needs are present, default to 0
            normalized_impacts = {}
            for need in NEED_NAMES:
                val = impacts.get(need, 0)
                try:
                    val = float(val)
                    val = max(-1.0, min(1.0, val))
                except (ValueError, TypeError):
                    val = 0.0
                normalized_impacts[need] = val

            verdict["impacts"] = normalized_impacts
            verdict.setdefault("narrative", "")
            verdict.setdefault("magnitude", "moderate")
            verdict["cached"] = False
            verdict["activity"] = activity  # for ledger reference

            # Store in cache (also auto-persists to disk)
            try:
                self._cache.store(cache_desc, embedding, {
                    "impacts": normalized_impacts,
                    "narrative": verdict["narrative"],
                    "magnitude": verdict["magnitude"],
                })
            except Exception as cache_err:
                import sys
                print(f"[judge] Cache store failed: {cache_err}", file=sys.stderr)

            return verdict

        except Exception as eval_err:
            import sys
            print(f"[judge] Evaluate failed: {eval_err}", file=sys.stderr)
            return None

    def apply_verdict(self, needs_state, verdict: Dict[str, Any]) -> None:
        """
        Apply a verdict's impacts to a NeedState object.

        The raw impact values (-1 to +1) are scaled by the magnitude to
        produce a per-step delta appropriate for the simulation's time scale.
        """
        if not verdict:
            return

        magnitude = verdict.get("magnitude", "moderate")
        scale = MAGNITUDE_SCALES.get(magnitude, MAGNITUDE_SCALES["moderate"])
        impacts = verdict.get("impacts", {})

        for need, raw_impact in impacts.items():
            if need in NEED_NAMES and raw_impact != 0:
                delta = raw_impact * scale
                needs_state.update(need, delta)

    @property
    def cache(self) -> JudgeCache:
        return self._cache


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_judge_instance: Optional[JudgeAgent] = None


def get_judge() -> JudgeAgent:
    """Get or create the global JudgeAgent singleton."""
    global _judge_instance
    if _judge_instance is None:
        persist_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "distill_logs", "judge_cache.json"
        )
        cache = JudgeCache(persist_path=persist_path)
        _judge_instance = JudgeAgent(cache=cache)
    return _judge_instance

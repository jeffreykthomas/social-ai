#!/usr/bin/env python3
"""
Generate life-plausible interaction scenarios for a simulation's personas.

Writes: environment/frontend_server/storage/<sim_code>/reverie/scenarios.json

Uses the existing ModelRouter ("teacher") if OPENAI_API_KEY is set; otherwise
falls back to a heuristic generator.

Scenarios are seeds: each has a default location and optional follow-up locations.
They can be used later by planning/invitation logic to expand interaction spaces.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load repo-root `.env` (if present) before we check OPENAI_API_KEY.
try:
    from reverie.backend_server.env import load_env as _load_env  # type: ignore
except Exception:
    _load_env = None

if _load_env:
    _load_env()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sim_dir(root: Path, sim_code: str) -> Path:
    return root / "environment" / "frontend_server" / "storage" / sim_code


def load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def list_places_from_spatial(spatial_memory: Dict[str, Any]) -> List[str]:
    world = spatial_memory.get("the Ville", {})
    if not isinstance(world, dict):
        return []
    # Places are top-level keys under "the Ville"
    return sorted([str(k) for k in world.keys() if k])


def read_persona_summaries(sim: Path) -> List[Dict[str, Any]]:
    meta = load_json(sim / "reverie" / "meta.json")
    persona_names = meta.get("persona_names") or []
    profiles_map = {}
    prof_file = sim / "reverie" / "persona_profiles.json"
    if prof_file.exists():
        profiles_map = load_json(prof_file) or {}

    summaries: List[Dict[str, Any]] = []
    for name in persona_names:
        scratch_p = sim / "personas" / name / "bootstrap_memory" / "scratch.json"
        scratch = load_json(scratch_p) if scratch_p.exists() else {}
        summaries.append(
            {
                "name": name,
                "profile_id": profiles_map.get(name) or scratch.get("persona_profile_id"),
                "age": scratch.get("age"),
                "innate": scratch.get("innate"),
                "learned": scratch.get("learned"),
                "currently": scratch.get("currently"),
                "daily_plan_req": scratch.get("daily_plan_req"),
                "living_area": scratch.get("living_area"),
            }
        )
    return summaries


def heuristic_scenarios(
    *,
    personas: List[Dict[str, Any]],
    places: List[str],
    rng: random.Random,
    count: int,
    max_group: int,
) -> List[Dict[str, Any]]:
    # Prefer high-traffic places if available.
    preferred = [
        "Hobbs Cafe",
        "The Willows Market and Pharmacy",
        "Harvey Oak Supply Store",
        "The Rose and Crown Pub",
        "Johnson Park",
        "Dorm for Oak Hill College",
    ]
    pick_places = [p for p in preferred if p in places] or places

    names = [p["name"] for p in personas]
    scenarios: List[Dict[str, Any]] = []
    for i in range(count):
        group_size = 2 if max_group <= 2 else rng.randint(2, max_group)
        participants = rng.sample(names, k=min(group_size, len(names)))
        loc = rng.choice(pick_places) if pick_places else "the Ville"
        followups = rng.sample(places, k=min(2, len(places))) if places else []
        scenarios.append(
            {
                "id": f"heur_{i+1:03d}",
                "participants": participants,
                "default_location": loc,
                "time_window": rng.choice(["morning", "midday", "afternoon", "evening"]),
                "premise": rng.choice(
                    [
                        "a routine errand overlaps",
                        "a work-related check-in happens in public",
                        "someone asks a small favor and it pulls others in",
                        "a chance encounter turns into a short conversation",
                    ]
                ),
                "rationale": "Heuristic seed scenario (no LLM available).",
                "followup_locations": followups[:2],
            }
        )
    return scenarios


def llm_scenarios(
    *,
    personas: List[Dict[str, Any]],
    places: List[str],
    rng_seed: int,
    count: int,
    max_group: int,
) -> List[Dict[str, Any]]:
    """
    Uses ModelRouter teacher. Requires OPENAI_API_KEY and openai package installed.
    """
    # Late imports so heuristic mode works without deps.
    from openai import OpenAI  # type: ignore

    # Wire API key the same way the backend does.
    from reverie.backend_server.utils import openai_api_key  # type: ignore
    from reverie.backend_server.models.router import get_router  # type: ignore

    client = OpenAI(api_key=openai_api_key)
    router = get_router(openai_client=client)

    # Keep the request bounded even if user asks for huge numbers.
    hard_cap = 200
    req_count = min(int(count), hard_cap)

    system = (
        "You are generating plausible real-life interaction scenarios for 9 simulated residents. "
        "Return STRICT JSON only (no markdown)."
    )

    user = {
        "task": "Generate scenario seeds where these agents might naturally see each other in daily life.",
        "requirements": {
            "scenario_count": req_count,
            "max_group_size": max_group,
            "locations_must_be_one_of": places,
            "each_scenario_fields": [
                "id",
                "participants",
                "default_location",
                "time_window",
                "premise",
                "rationale",
                "followup_locations",
            ],
            "constraints": [
                "participants must be a subset of the provided persona names",
                "default_location must be exactly one of the allowed locations",
                "followup_locations must be a list of allowed locations (0-3 items)",
                "premise should be concrete (bank/grocery/work/school dropoff style), not abstract",
                "rationale should explain why these specific people plausibly overlap",
            ],
        },
        "personas": personas,
        "random_seed": rng_seed,
        "output_format": {
            "scenarios": [
                {
                    "id": "scn_001",
                    "participants": ["Name A", "Name B"],
                    "default_location": "Hobbs Cafe",
                    "time_window": "morning|midday|afternoon|evening",
                    "premise": "short description of what brings them together",
                    "rationale": "why these personas overlap given their lives",
                    "followup_locations": ["Johnson Park", "The Willows Market and Pharmacy"],
                }
            ]
        },
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]

    msg, _debug = router.chat(
        provider="teacher",
        task="scenarios",
        messages=messages,
        temperature=0.4,
        max_tokens=1400,
        meta={"purpose": "scenario_generation", "seed": rng_seed, "count": req_count},
    )

    content = (msg.get("content") or "").strip()
    data = json.loads(content)
    scenarios = data.get("scenarios") or []
    if not isinstance(scenarios, list):
        raise ValueError("LLM returned invalid schema (scenarios not a list)")

    # Basic validation and cleanup
    allowed_names = {p["name"] for p in personas}
    allowed_places = set(places)
    cleaned: List[Dict[str, Any]] = []
    for i, s in enumerate(scenarios[:req_count]):
        if not isinstance(s, dict):
            continue
        participants = [x for x in (s.get("participants") or []) if x in allowed_names]
        if len(participants) < 2:
            continue
        loc = s.get("default_location")
        if loc not in allowed_places:
            continue
        followups = [x for x in (s.get("followup_locations") or []) if x in allowed_places and x != loc]
        cleaned.append(
            {
                "id": s.get("id") or f"scn_{i+1:03d}",
                "participants": participants[: max_group],
                "default_location": loc,
                "time_window": s.get("time_window") or "midday",
                "premise": s.get("premise") or "",
                "rationale": s.get("rationale") or "",
                "followup_locations": followups[:3],
            }
        )

    return cleaned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="Simulation folder name under environment/frontend_server/storage")
    ap.add_argument("--count", type=int, default=36, help="How many scenarios to generate (default 36)")
    ap.add_argument("--max-group", type=int, default=3, help="Max participants per scenario (2-5 recommended)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 means derive from current time)")
    args = ap.parse_args()

    root = repo_root()
    sim = sim_dir(root, args.sim)
    if not sim.exists():
        raise FileNotFoundError(f"Simulation not found: {sim}")

    seed = args.seed if args.seed != 0 else int(datetime.now().timestamp())
    rng = random.Random(seed)

    # Pull a reference spatial memory from any persona folder.
    spatial_memory = None
    for p in (sim / "personas").iterdir():
        sm = p / "bootstrap_memory" / "spatial_memory.json"
        if sm.exists():
            spatial_memory = load_json(sm)
            break
    if spatial_memory is None:
        raise FileNotFoundError("Could not find spatial_memory.json in this simulation")

    places = list_places_from_spatial(spatial_memory)
    personas = read_persona_summaries(sim)

    max_group = max(2, min(int(args.max_group), 6))
    count = max(1, int(args.count))

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    scenarios: List[Dict[str, Any]]
    if has_key:
        scenarios = llm_scenarios(personas=personas, places=places, rng_seed=seed, count=count, max_group=max_group)
        if not scenarios:
            # fallback if model output couldn't be parsed/validated
            scenarios = heuristic_scenarios(personas=personas, places=places, rng=rng, count=min(count, 36), max_group=max_group)
    else:
        scenarios = heuristic_scenarios(personas=personas, places=places, rng=rng, count=min(count, 36), max_group=max_group)

    out = {
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "count_requested": count,
        "count_written": len(scenarios),
        "max_group": max_group,
        "scenarios": scenarios,
    }
    dump_json(sim / "reverie" / "scenarios.json", out)
    print(f"Wrote {len(scenarios)} scenarios to {sim/'reverie'/'scenarios.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



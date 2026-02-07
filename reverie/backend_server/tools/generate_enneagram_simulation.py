#!/usr/bin/env python3
"""
Generate a fresh 9-agent simulation (storage folder) whose agents are seeded from
Enneagram-inspired persona profiles.

Design goals:
- 9 different agents each run (names/scratch vary)
- foundational similarity per Enneagram type comes from profiles/enneagram.yaml
- compatible with existing Reverie storage layout:
    environment/0.json, reverie/meta.json, personas/<Name>/bootstrap_memory/...

Usage (run from repo root):
  python reverie/backend_server/tools/generate_enneagram_simulation.py \
    --base base_the_ville_n25 \
    --out enneagram9_run1

To also generate interaction scenarios (requires OPENAI_API_KEY for LLM mode):
  python reverie/backend_server/tools/generate_enneagram_simulation.py \
    --base base_the_ville_n25 \
    --out enneagram9_run1 \
    --scenarios 36 --max-group 3

After generating, you can run the normal simulator fork flow using the new folder
as the fork simulation code.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def repo_root() -> Path:
    # .../reverie/backend_server/tools/this_file.py -> root is parents[3]
    return Path(__file__).resolve().parents[3]


def storage_dir(root: Path) -> Path:
    return root / "environment" / "frontend_server" / "storage"


def load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def read_profiles(root: Path) -> Dict[str, Dict[str, Any]]:
    p = root / "reverie" / "backend_server" / "persona" / "profiles" / "enneagram.yaml"
    data = yaml.safe_load(p.read_text()) or {}
    profiles = data.get("profiles", {}) or {}
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be a mapping in enneagram.yaml")
    return {k: v for k, v in profiles.items() if isinstance(v, dict)}


def pick_seed_spawns(env0: Dict[str, Any], n: int, rng: random.Random) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for _, v in env0.items():
        try:
            coords.append((int(v["x"]), int(v["y"])))
        except Exception:
            continue
    if len(coords) < n:
        raise ValueError(f"Base environment has only {len(coords)} spawn points; need {n}")
    return rng.sample(coords, n)


def load_reference_spatial_memory(base_sim: Path) -> Dict[str, Any]:
    personas_dir = base_sim / "personas"
    if not personas_dir.exists():
        raise FileNotFoundError(f"Expected personas folder: {personas_dir}")
    # Use first persona's spatial memory as a world reference (they are typically identical).
    for child in personas_dir.iterdir():
        sm = child / "bootstrap_memory" / "spatial_memory.json"
        if sm.exists():
            return load_json(sm)
    raise FileNotFoundError(f"No spatial_memory.json found under {personas_dir}")


def residence_pool(spatial_memory: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Returns list of (place_name, arena_name) pairs suitable for Scratch.living_area.
    """
    world = spatial_memory.get("the Ville", {})
    if not isinstance(world, dict):
        return []
    candidates: List[Tuple[str, str]] = []
    for place_name, sectors in world.items():
        if not isinstance(sectors, dict):
            continue
        lname = str(place_name).lower()
        if "apartment" not in lname and "home" not in lname and "house" not in lname and "dorm" not in lname:
            continue
        # pick first arena available within that place
        for arena_name in sectors.keys():
            if arena_name:
                candidates.append((str(place_name), str(arena_name)))
                break
    return candidates


FIRST_NAMES = [
    "Ava", "Noah", "Maya", "Ethan", "Lina", "Owen", "Sofia", "Lucas", "Zara",
    "Amir", "Nina", "Leo", "Iris", "Mateo", "Claire", "Jasper", "Priya", "Theo",
]

LAST_NAMES = [
    "Reed", "Patel", "Nguyen", "Kim", "Hughes", "Morales", "Chen", "Singh",
    "Bennett", "Rossi", "Lopez", "Khan", "Park", "Johnson", "Garcia", "Taylor",
]


def unique_full_name(used: set, rng: random.Random) -> str:
    for _ in range(1000):
        fn = rng.choice(FIRST_NAMES)
        ln = rng.choice(LAST_NAMES)
        full = f"{fn} {ln}"
        if full not in used:
            used.add(full)
            return full
    raise RuntimeError("Failed to generate unique names")


def type_job_template(type_id: int, full_name: str, rng: random.Random) -> Tuple[str, str, str]:
    """
    Returns (daily_plan_req, learned, currently) strings.
    Keep locations tied to known map venues so the text stays grounded.
    """
    venues = [
        "Hobbs Cafe",
        "The Rose and Crown Pub",
        "Harvey Oak Supply Store",
        "The Willows Market and Pharmacy",
        "Dorm for Oak Hill College",
        "Johnson Park",
    ]
    venue = rng.choice(venues)

    # Small per-type pools; these create “family resemblance” across runs.
    pools = {
        1: [
            ("quality manager", "sets standards and keeps things orderly"),
            ("operations coordinator", "fixes process issues and keeps commitments"),
        ],
        2: [
            ("community volunteer", "checks in on people and offers practical help"),
            ("customer care associate", "remembers preferences and makes people feel seen"),
        ],
        3: [
            ("project lead", "optimizes outcomes and tracks progress visibly"),
            ("sales manager", "sets ambitious targets and rallies others to hit them"),
        ],
        4: [
            ("independent artist", "seeks authenticity and emotional depth"),
            ("writer", "turns lived experience into meaning and narrative"),
        ],
        5: [
            ("research assistant", "prefers deep work, analysis, and quiet competence"),
            ("systems analyst", "collects facts before acting and values privacy"),
        ],
        6: [
            ("safety coordinator", "anticipates risks and builds contingency plans"),
            ("logistics planner", "double-checks details and relies on trusted routines"),
        ],
        7: [
            ("events promoter", "creates fun plans and keeps options open"),
            ("tour guide", "loves novelty and pulls people into shared experiences"),
        ],
        8: [
            ("site supervisor", "takes charge, protects the group, and moves decisively"),
            ("small business owner", "values independence and hates being controlled"),
        ],
        9: [
            ("front-desk attendant", "keeps things calm and helps people get along"),
            ("community mediator", "smooths tensions and avoids unnecessary conflict"),
        ],
    }
    role, flavor = rng.choice(pools[type_id])

    daily = (
        f"{full_name} works as a {role} centered around {venue}. "
        f"They keep a steady routine from 9am to 6pm, with a short break around 1pm."
    )
    learned = (
        f"{full_name} is known as a {role} who {flavor}. "
        f"They approach daily life with a consistent style that others notice quickly."
    )
    currently = (
        f"{full_name} is currently focused on a small but meaningful goal this week: "
        f"making one interaction at {venue} go especially well, without forcing it."
    )
    return daily, learned, currently


def type_innate(type_id: int) -> str:
    traits = {
        1: "principled, conscientious, improvement-oriented",
        2: "warm, helpful, attentive",
        3: "ambitious, polished, outcome-focused",
        4: "expressive, introspective, authenticity-driven",
        5: "observant, analytical, private",
        6: "loyal, cautious, preparedness-focused",
        7: "optimistic, playful, novelty-seeking",
        8: "assertive, protective, independent",
        9: "steady, agreeable, harmony-seeking",
    }
    return traits[type_id]


def build_scratch(full_name: str, profile_id: str, type_id: int, spatial_residence: Tuple[str, str], rng: random.Random) -> Dict[str, Any]:
    first, last = full_name.split(" ", 1)
    age = rng.randint(24, 52)

    daily_plan_req, learned, currently = type_job_template(type_id, full_name, rng)
    innate = type_innate(type_id)

    place_name, arena_name = spatial_residence
    living_area = f"the Ville:{place_name}:{arena_name}"

    # Keep these consistent with existing scratch schema.
    scratch: Dict[str, Any] = {
        "vision_r": 8,
        "att_bandwidth": 8,
        "retention": 8,
        "curr_time": None,
        "curr_tile": None,
        "daily_plan_req": daily_plan_req,
        "name": full_name,
        "first_name": first,
        "last_name": last,
        "age": age,
        "innate": innate,
        "learned": learned,
        "currently": currently,
        "lifestyle": f"{full_name} goes to bed around {rng.choice(['10pm','11pm','12am'])}, and wakes up around {rng.choice(['6am','7am','8am'])}.",
        "living_area": living_area,
        "concept_forget": 100,
        "daily_reflection_time": 180,
        "daily_reflection_size": 5,
        "overlap_reflect_th": 4,
        "kw_strg_event_reflect_th": 10,
        "kw_strg_thought_reflect_th": 9,
        "recency_w": 1,
        "relevance_w": 1,
        "importance_w": 1,
        "recency_decay": 0.995,
        "importance_trigger_max": 150,
        "importance_trigger_curr": 150,
        "importance_ele_n": 0,
        "thought_count": 5,
        "daily_req": [],
        "f_daily_schedule": [],
        "f_daily_schedule_hourly_org": [],
        "act_address": None,
        "act_start_time": None,
        "act_duration": None,
        "act_description": None,
        "act_pronunciatio": None,
        "act_event": [full_name, None, None],
        "act_obj_description": None,
        "act_obj_pronunciatio": None,
        "act_obj_event": [None, None, None],
        "chatting_with": None,
        "chat": None,
        "chatting_with_buffer": {},
        "chatting_end_time": None,
        "act_path_set": False,
        "planned_path": [],
        # Extra debugging/traceability fields (safe: Scratch loader ignores unknown keys)
        "persona_profile_id": profile_id,
        "persona_type": type_id,
    }
    return scratch


def write_persona_folder(sim_out: Path, full_name: str, spatial_memory: Dict[str, Any], scratch: Dict[str, Any]) -> None:
    base = sim_out / "personas" / full_name / "bootstrap_memory"
    (base / "associative_memory").mkdir(parents=True, exist_ok=True)

    dump_json(base / "scratch.json", scratch)
    dump_json(base / "spatial_memory.json", spatial_memory)
    dump_json(base / "associative_memory" / "nodes.json", {})
    dump_json(base / "associative_memory" / "embeddings.json", {})
    dump_json(base / "associative_memory" / "kw_strength.json", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base_the_ville_n25", help="Base simulation folder to draw spawn points/world memory from")
    ap.add_argument("--out", default="", help="Output simulation folder name (sim_code). Defaults to timestamped name.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 means derive from current time)")
    ap.add_argument("--scenarios", type=int, default=36, help="Generate this many scenario seeds and write reverie/scenarios.json (default 36)")
    ap.add_argument("--max-group", type=int, default=3, help="Max participants per scenario seed (default 3)")
    args = ap.parse_args()

    root = repo_root()
    store = storage_dir(root)
    base_sim = store / args.base
    if not base_sim.exists():
        raise FileNotFoundError(f"Base simulation not found: {base_sim}")

    seed = args.seed if args.seed != 0 else int(datetime.now().timestamp())
    rng = random.Random(seed)

    sim_code = args.out.strip() or f"enneagram9_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sim_out = store / sim_code
    if sim_out.exists():
        raise FileExistsError(f"Output simulation already exists: {sim_out}")

    profiles = read_profiles(root)
    profile_ids = [f"enneagram_{i}" for i in range(1, 10)]
    for pid in profile_ids:
        if pid not in profiles:
            raise ValueError(f"Missing profile id '{pid}' in enneagram.yaml")

    env0 = load_json(base_sim / "environment" / "0.json")
    spawns = pick_seed_spawns(env0, 9, rng)

    spatial = load_reference_spatial_memory(base_sim)
    residences = residence_pool(spatial)
    if not residences:
        raise RuntimeError("Could not find any residences in spatial memory to use for living_area")

    used_names: set = set()
    persona_names: List[str] = []
    env_out: Dict[str, Any] = {}
    profiles_map: Dict[str, str] = {}

    for idx, type_id in enumerate(range(1, 10)):
        profile_id = f"enneagram_{type_id}"
        full = unique_full_name(used_names, rng)
        persona_names.append(full)
        profiles_map[full] = profile_id

        res = rng.choice(residences)
        scratch = build_scratch(full, profile_id, type_id, res, rng)
        write_persona_folder(sim_out, full, spatial, scratch)

        x, y = spawns[idx]
        env_out[full] = {"maze": "the_ville", "x": x, "y": y}

    # Write environment and meta
    dump_json(sim_out / "environment" / "0.json", env_out)

    base_meta = load_json(base_sim / "reverie" / "meta.json")
    meta = dict(base_meta)
    meta["fork_sim_code"] = args.base
    meta["persona_names"] = persona_names
    meta["step"] = 0
    dump_json(sim_out / "reverie" / "meta.json", meta)

    # Helpful mapping for downstream (e.g., PersonaManager can use it)
    dump_json(sim_out / "reverie" / "persona_profiles.json", profiles_map)

    # Generate scenario seeds (LLM-backed if OPENAI_API_KEY is set; else heuristic).
    try:
        from reverie.backend_server.tools.generate_scenarios_for_sim import main as _gen_scenarios_main  # type: ignore
        # Run generator by sim code, sharing the same seed so runs are reproducible.
        # We call it via its CLI entry to keep logic in one place.
        import sys as _sys
        argv_old = list(_sys.argv)
        _sys.argv = [
            argv_old[0],
            "--sim",
            sim_code,
            "--count",
            str(max(1, int(args.scenarios))),
            "--max-group",
            str(max(2, int(args.max_group))),
            "--seed",
            str(seed),
        ]
        _gen_scenarios_main()
        _sys.argv = argv_old
    except Exception as e:
        print(f"(warning) Scenario generation skipped/failed: {e}")

    print(f"Generated simulation: {sim_code}")
    print(f"- Folder: {sim_out}")
    print(f"- Seed: {seed}")
    print(f"- Agents: {', '.join(persona_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



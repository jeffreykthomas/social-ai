"""
Persona Profile Loader (top-level package)
-----------------------------------------
Keeps the same API as `persona/profiles/profile_loader.py`:
  - load_profiles()
  - get_profile()
  - profile_style_text()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_CACHE: Dict[str, Dict[str, Any]] = {}


def _default_profiles_path() -> Path:
    return Path(__file__).parent / "enneagram.yaml"


def load_profiles(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    p = Path(path) if path else _default_profiles_path()
    key = str(p.resolve())
    if key in _CACHE:
        return _CACHE[key]

    if not p.exists():
        raise FileNotFoundError(f"Persona profiles file not found: {p}")

    data = yaml.safe_load(p.read_text()) or {}
    profiles = data.get("profiles", {}) or {}
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be a mapping in the YAML file")

    normalized: Dict[str, Dict[str, Any]] = {}
    for profile_id, prof in profiles.items():
        if not isinstance(prof, dict):
            continue
        prof = {**prof}
        prof.setdefault("id", profile_id)
        normalized[profile_id] = prof

    _CACHE[key] = normalized
    return normalized


def get_profile(profile_id: str, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    profiles = load_profiles(path)
    return profiles.get(profile_id)


def profile_style_text(profile: Optional[Dict[str, Any]]) -> str:
    if not profile:
        return ""

    parts = []
    label = profile.get("label") or profile.get("name")
    if label:
        parts.append(f"Persona archetype: {label}")

    core = profile.get("core_motive")
    if core:
        parts.append(f"Core motive: {core}")

    fear = profile.get("core_fear")
    if fear:
        parts.append(f"Core fear: {fear}")

    comm = profile.get("communication_style") or {}
    if isinstance(comm, dict):
        tone = comm.get("tone")
        if tone:
            parts.append(f"Tone: {tone}")
        directness = comm.get("directness")
        if directness:
            parts.append(f"Directness: {directness}")
        affect = comm.get("emotional_expression")
        if affect:
            parts.append(f"Emotional expression: {affect}")
        conflict = comm.get("conflict_style")
        if conflict:
            parts.append(f"Conflict style: {conflict}")

    return "\n".join(f"- {p}" for p in parts[:4])



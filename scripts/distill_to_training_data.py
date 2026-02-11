#!/usr/bin/env python3
"""
Convert teacher distillation logs (teacher.jsonl) into chat-format training
data that can be used to fine-tune the student model (Qwen2.5) via SFT.

Input:  model/training/arena/teacher.jsonl (falls back to reverie path if present)
Output: model/training/arena/training/  (train.jsonl, val.jsonl)

Each output line is:
  {"messages": [{"role":"system","content":"..."},
                {"role":"user","content":"..."},
                {"role":"assistant","content":"..."}],
   "task": "schedule",
   "tools": [...] | null}

For tool-call responses the assistant message is serialised as:
  {"role": "assistant", "content": null,
   "tool_calls": [{"type":"function","function":{"name":"...","arguments":"..."}}]}

This is the same schema used by OpenAI fine-tuning and understood by TRL's
chat-template SFT trainer with Qwen's chat template.

Usage:
  python scripts/distill_to_training_data.py [OPTIONS]

  --input   Path to teacher.jsonl (default: auto-detected)
  --outdir  Output directory (default: distill_logs/training/)
  --split   Validation split fraction (default: 0.1)
  --min-content-len  Discard entries whose combined content is shorter (default: 20)
  --tasks   Comma-separated task filter (default: all tasks)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_input() -> Path:
    social_pet = repo_root() / "model" / "training" / "arena" / "teacher.jsonl"
    if social_pet.exists():
        return social_pet
    return repo_root() / "reverie" / "backend_server" / "distill_logs" / "teacher.jsonl"


def default_outdir() -> Path:
    return repo_root() / "model" / "training" / "arena" / "training"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _sanitize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """
    Normalise tool_calls from the OpenAI response format into the minimal
    schema needed for training:
      [{"type": "function", "function": {"name": "...", "arguments": "..."}}]
    """
    out = []
    if not isinstance(tool_calls, list):
        return out
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments", "{}")
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        out.append({
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    return out


def convert_entry(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a single teacher.jsonl record into a training example.
    Returns None if the entry should be skipped.
    """
    messages_in: List[Dict[str, Any]] = rec.get("messages") or []
    response: Any = rec.get("response")
    task: str = rec.get("task", "unknown")
    tools: Any = rec.get("tools")

    if not messages_in or not response:
        return None

    # Build the conversation: input messages + assistant response.
    conv: List[Dict[str, Any]] = []
    for m in messages_in:
        role = m.get("role", "user")
        content = m.get("content", "")
        conv.append({"role": role, "content": content})

    # Build assistant turn from the response.
    if isinstance(response, dict):
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        tool_calls = _sanitize_tool_calls(response.get("tool_calls"))
        content = response.get("content")
        if tool_calls:
            # Tool-call response: content is typically null/empty.
            assistant_msg["content"] = content or None
            assistant_msg["tool_calls"] = tool_calls
        else:
            assistant_msg["content"] = content or ""
        conv.append(assistant_msg)
    elif isinstance(response, str):
        conv.append({"role": "assistant", "content": response})
    else:
        return None

    out: Dict[str, Any] = {"messages": conv, "task": task}
    if tools:
        out["tools"] = tools
    tool_choice = rec.get("tool_choice")
    if tool_choice:
        out["tool_choice"] = tool_choice
    return out


def content_length(example: Dict[str, Any]) -> int:
    """Total character length of all message content in an example."""
    total = 0
    for m in example.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(fn.get("arguments", ""))
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Convert teacher distillation logs to training data")
    ap.add_argument("--input", type=str, default=None,
                    help="Path to teacher.jsonl")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory for train.jsonl / val.jsonl")
    ap.add_argument("--split", type=float, default=0.1,
                    help="Fraction of data to hold out for validation (default 0.1)")
    ap.add_argument("--min-content-len", type=int, default=20,
                    help="Discard examples with less total content (default 20)")
    ap.add_argument("--tasks", type=str, default="",
                    help="Comma-separated task filter (empty = all)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/val split")
    args = ap.parse_args()

    input_path = Path(args.input) if args.input else default_input()
    outdir = Path(args.outdir) if args.outdir else default_outdir()
    task_filter = set(t.strip() for t in args.tasks.split(",") if t.strip()) if args.tasks else None

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Read and convert
    examples: List[Dict[str, Any]] = []
    skipped = 0
    task_counts: Dict[str, int] = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARNING: skipping malformed JSON at line {lineno}", file=sys.stderr)
                skipped += 1
                continue

            task = rec.get("task", "unknown")
            if task_filter and task not in task_filter:
                continue

            example = convert_entry(rec)
            if example is None:
                skipped += 1
                continue

            if content_length(example) < args.min_content_len:
                skipped += 1
                continue

            examples.append(example)
            task_counts[task] = task_counts.get(task, 0) + 1

    if not examples:
        print("ERROR: No valid training examples produced.", file=sys.stderr)
        return 1

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(examples)

    val_size = max(1, int(len(examples) * args.split)) if len(examples) > 3 else 0
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    # Write output
    outdir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = outdir / "train.jsonl"
    val_path = outdir / "val.jsonl"
    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    # Summary
    print(f"Input:       {input_path}")
    print(f"Output dir:  {outdir}")
    print(f"Total entries read:   {len(examples) + skipped}")
    print(f"Valid examples:       {len(examples)}")
    print(f"Skipped:              {skipped}")
    print(f"Task distribution:    {task_counts}")
    print(f"Train examples:       {len(train_examples)}  -> {train_path}")
    print(f"Val examples:         {len(val_examples)}  -> {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

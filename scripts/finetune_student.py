#!/usr/bin/env python3
"""
LoRA SFT fine-tuning for the student model using teacher distillation data.

Workflow:
  1. Convert teacher.jsonl → chat-format training examples (reuses distill_to_training_data logic)
  2. Apply LoRA adapters to the base student model
  3. Train via Hugging Face SFTTrainer
  4. Save adapter + optionally merge into a full model

Usage:
  python scripts/finetune_student.py [OPTIONS]

  --base-model      HF model id or local path (default: Qwen/Qwen2.5-32B-Instruct)
  --teacher-log     Path to teacher.jsonl (default: auto-detected)
  --output-dir      Where to save the adapter (default: models/student_lora/adapter)
  --merge           Also save a fully-merged model to models/student_lora/merged
  --epochs          Number of training epochs (default: 2)
  --batch-size      Per-device batch size (default: 1)
  --grad-accum      Gradient accumulation steps (default: 8)
  --lr              Learning rate (default: 2e-4)
  --lora-rank       LoRA rank (default: 32)
  --lora-alpha      LoRA alpha (default: 64)
  --max-seq-len     Max sequence length for training (default: 4096)
  --min-examples    Minimum training examples required to proceed (default: 100)
  --bf16            Use bfloat16 training (default: true on supported hardware)

Requires:
  pip install "trl>=0.16" "peft>=0.14" transformers torch accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_teacher_log() -> Path:
    social_pet = repo_root() / "model" / "training" / "arena" / "teacher.jsonl"
    if social_pet.exists():
        return social_pet
    return repo_root() / "reverie" / "backend_server" / "distill_logs" / "teacher.jsonl"


def default_output_dir() -> Path:
    return repo_root() / "models" / "student_lora" / "adapter"


def default_merged_dir() -> Path:
    return repo_root() / "models" / "student_lora" / "merged"


# ---------------------------------------------------------------------------
# Data conversion (inline, mirrors distill_to_training_data.py logic)
# ---------------------------------------------------------------------------

def convert_teacher_entry(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single teacher.jsonl record into a chat training example."""
    messages_in: List[Dict[str, Any]] = rec.get("messages") or []
    response: Any = rec.get("response")
    task: str = rec.get("task", "unknown")
    tools: Any = rec.get("tools")

    if not messages_in or not response:
        return None

    conv: List[Dict[str, Any]] = []
    for m in messages_in:
        role = m.get("role", "user")
        content = m.get("content", "")
        conv.append({"role": role, "content": content})

    # Build assistant turn
    if isinstance(response, dict):
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        tool_calls = _sanitize_tool_calls(response.get("tool_calls"))
        content = response.get("content")
        if tool_calls:
            # For tool-call responses, encode the tool call as the assistant content
            # in a format the chat template understands.
            tc_text = json.dumps(tool_calls, ensure_ascii=False)
            assistant_msg["content"] = content or ""
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


def _sanitize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
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


def _content_length(example: Dict[str, Any]) -> int:
    total = 0
    for m in example.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(fn.get("arguments", ""))
    return total


def load_training_examples(
    teacher_log: Path,
    min_content_len: int = 20,
    task_filter: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Load and convert teacher.jsonl into training examples."""
    examples = []
    skipped = 0

    with open(teacher_log, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            task = rec.get("task", "unknown")
            if task_filter and task not in task_filter:
                continue

            example = convert_teacher_entry(rec)
            if example is None:
                skipped += 1
                continue

            if _content_length(example) < min_content_len:
                skipped += 1
                continue

            examples.append(example)

    print(f"Loaded {len(examples)} training examples ({skipped} skipped) from {teacher_log}")
    return examples


# ---------------------------------------------------------------------------
# Formatting for SFTTrainer
# ---------------------------------------------------------------------------

def format_example_for_sft(example: Dict[str, Any]) -> str:
    """
    Format a training example as a single string for SFTTrainer.

    For tool-call examples, we encode the tool call in the assistant message
    using Qwen/Hermes function-calling format so the model learns the correct
    structured output pattern.
    """
    parts = []
    for msg in example.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if role == "system":
            # Include tools in system message if present
            tools = example.get("tools")
            if tools:
                tools_text = json.dumps(tools, indent=2, ensure_ascii=False)
                content = f"{content}\n\nAvailable tools:\n{tools_text}"
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            if tool_calls:
                # Encode tool calls in Hermes format
                tc_parts = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.dumps(json.loads(args), ensure_ascii=False)
                        except Exception:
                            pass
                    tc_parts.append(
                        f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'
                    )
                assistant_content = "\n".join(tc_parts)
                if content:
                    assistant_content = f"{content}\n{assistant_content}"
                parts.append(f"<|im_start|>assistant\n{assistant_content}<|im_end|>")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA SFT fine-tune student on teacher distillation data")
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--teacher-log", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--merge", action="store_true", help="Also save a fully-merged model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--min-examples", type=int, default=100)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--tasks", type=str, default="", help="Comma-separated task filter (empty = all)")
    args = ap.parse_args()

    teacher_log = Path(args.teacher_log) if args.teacher_log else default_teacher_log()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    use_bf16 = args.bf16 and not args.no_bf16

    if not teacher_log.exists():
        print(f"ERROR: Teacher log not found: {teacher_log}", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    task_filter = set(t.strip() for t in args.tasks.split(",") if t.strip()) if args.tasks else None
    examples = load_training_examples(teacher_log, task_filter=task_filter)

    if len(examples) < args.min_examples:
        print(
            f"Only {len(examples)} examples (minimum {args.min_examples}). "
            f"Skipping training — accumulate more teacher data first."
        )
        return 2  # exit code 2 = not enough data (watcher can check this)

    # Format for SFT
    formatted = [format_example_for_sft(ex) for ex in examples]
    print(f"Formatted {len(formatted)} examples for SFT training")

    # -----------------------------------------------------------------------
    # 2. Load model + apply LoRA
    # -----------------------------------------------------------------------
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Detect if we're running under accelerate (FSDP multi-GPU) or standalone.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_main = local_rank == 0

    if is_main:
        print(f"\nTraining mode: {'FSDP multi-GPU' if is_distributed else 'single-GPU'} "
              f"(world_size={world_size})")
        print(f"Loading base model: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_distributed:
        # FSDP mode: load in bf16 on CPU, let FSDP shard to GPUs.
        # low_cpu_mem_usage keeps params on CPU meta; FSDP moves shards to GPUs.
        # 4x 48GB GPUs = 192GB total, 32B model in bf16 = ~64GB → plenty of room.
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        # Single-GPU fallback: QLoRA (4-bit quantized).
        from transformers import BitsAndBytesConfig
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )

    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # -----------------------------------------------------------------------
    # 3. Build dataset
    # -----------------------------------------------------------------------
    dataset = Dataset.from_dict({"text": formatted})

    # 90/10 split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # -----------------------------------------------------------------------
    # 4. Train
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        max_length=args.max_seq_len,
        dataset_text_field="text",
        packing=True,  # slight cross-contamination without flash_attn, acceptable for tool-call SFT
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",  # no wandb/tensorboard for now
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"\nStarting LoRA SFT training ({args.epochs} epochs, rank={args.lora_rank})...")
    trainer.train()

    # -----------------------------------------------------------------------
    # 5. Save
    # -----------------------------------------------------------------------
    print(f"\nSaving LoRA adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Write a metadata file for the watcher
    meta = {
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "num_examples": len(examples),
        "epochs": args.epochs,
        "teacher_log": str(teacher_log),
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

    if args.merge:
        merged_dir = default_merged_dir()
        print(f"Merging adapter into full model at {merged_dir}...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Merged model saved to {merged_dir}")

    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

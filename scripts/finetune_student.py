#!/usr/bin/env python3
"""
Fine-tune the student model (Qwen2.5) on distilled teacher data using LoRA.

This script:
1. Loads chat-format JSONL produced by distill_to_training_data.py
2. Applies Qwen's chat template to format each conversation
3. Trains a LoRA adapter via PEFT + TRL SFTTrainer
4. Saves the adapter (and optionally merges into a full model)

The resulting adapter can be served by vLLM with --enable-lora, or merged
into the base model for standalone serving.

Usage:
  python scripts/finetune_student.py [OPTIONS]

  --base-model   HuggingFace model id (default: openai/gpt-oss-120b)
  --train-data   Path to train.jsonl (default: distill_logs/training/train.jsonl)
  --val-data     Path to val.jsonl   (default: distill_logs/training/val.jsonl)
  --output-dir   Where to save checkpoints (default: models/student_lora/)
  --merge        If set, merge LoRA into base and save full model
  --epochs       Number of training epochs (default: 3)
  --lr           Learning rate (default: 2e-4)
  --lora-r       LoRA rank (default: 32)
  --lora-alpha   LoRA alpha (default: 64)
  --max-seq-len  Max sequence length (default: 4096)
  --batch-size   Per-device train batch size (default: 1)
  --grad-accum   Gradient accumulation steps (default: 8)

Note on model size:
  The default is GPT-OSS-120B (OpenAI open-weight MoE, 5.1B active params).
  This fits on a single GPU with 4-bit quant and trains quickly.
  For the smaller 20B serving model, set --base-model openai/gpt-oss-20b
  (3.6B active, fits on a single GPU with 4-bit quant and trains quickly).
  LoRA adapters are architecture-specific -- train on the same base you serve.
  
  To fall back to Qwen: --base-model Qwen/Qwen2.5-32B-Instruct

Requires:
  pip install torch transformers peft trl datasets bitsandbytes accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_train_data() -> Path:
    return repo_root() / "reverie" / "backend_server" / "distill_logs" / "training" / "train.jsonl"


def default_val_data() -> Path:
    return repo_root() / "reverie" / "backend_server" / "distill_logs" / "training" / "val.jsonl"


def default_output_dir() -> Path:
    return repo_root() / "models" / "student_lora"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_example_for_sft(example: Dict[str, Any], tokenizer) -> str:
    """
    Apply the model's chat template to a training example.

    The example has {"messages": [...], "tools": [...] | absent}.
    Qwen2.5-Instruct's tokenizer.apply_chat_template handles tool-call
    messages natively when tools are provided.
    """
    messages = example["messages"]
    tools = example.get("tools")

    try:
        # apply_chat_template returns a string with special tokens.
        # tokenize=False gives us the raw text for SFTTrainer.
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text
    except Exception as e:
        # Fallback: manual formatting if the template doesn't support tools.
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls")
            if tool_calls:
                # Serialize tool calls into the content.
                tc_str = json.dumps(tool_calls, ensure_ascii=False)
                parts.append(f"<|im_start|>{role}\n{tc_str}<|im_end|>")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune student model on distilled teacher data")
    ap.add_argument("--base-model", type=str,
                    default=os.environ.get("STUDENT_BASE_MODEL", "openai/gpt-oss-120b"),
                    help="HuggingFace model id for the student base")
    ap.add_argument("--train-data", type=str, default=None)
    ap.add_argument("--val-data", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--merge", action="store_true",
                    help="Merge LoRA adapter into base model and save full model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--load-in-4bit", action="store_true", default=True,
                    help="Load base model in 4-bit (QLoRA)")
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load data and print stats without training")
    args = ap.parse_args()

    train_path = Path(args.train_data) if args.train_data else default_train_data()
    val_path = Path(args.val_data) if args.val_data else default_val_data()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()

    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}", file=sys.stderr)
        print("  Run scripts/distill_to_training_data.py first.", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print(f"Loading training data from {train_path} ...")
    train_raw = load_jsonl(train_path)
    val_raw = load_jsonl(val_path) if val_path.exists() else []

    print(f"  Train examples: {len(train_raw)}")
    print(f"  Val examples:   {len(val_raw)}")

    if not train_raw:
        print("ERROR: No training examples found.", file=sys.stderr)
        return 1

    # -----------------------------------------------------------------------
    # Load tokenizer (needed for chat template formatting)
    # -----------------------------------------------------------------------
    print(f"Loading tokenizer for {args.base_model} ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Format into text via chat template
    # -----------------------------------------------------------------------
    print("Applying chat template ...")
    train_texts = []
    for ex in train_raw:
        text = format_example_for_sft(ex, tokenizer)
        train_texts.append({"text": text})

    val_texts = []
    for ex in val_raw:
        text = format_example_for_sft(ex, tokenizer)
        val_texts.append({"text": text})

    # Token length stats
    lengths = [len(tokenizer.encode(t["text"])) for t in train_texts]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        print(f"  Avg tokens/example: {avg_len:.0f}")
        print(f"  Max tokens/example: {max_len}")
        if max_len > args.max_seq_len:
            print(f"  WARNING: {sum(1 for l in lengths if l > args.max_seq_len)} examples "
                  f"exceed --max-seq-len {args.max_seq_len} and will be truncated.")

    if args.dry_run:
        print("\n--- DRY RUN: showing first formatted example ---")
        if train_texts:
            print(train_texts[0]["text"][:2000])
        print("\nDry run complete. No training performed.")
        return 0

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"Loading base model {args.base_model} ...")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # LoRA config
    # -----------------------------------------------------------------------
    from peft import LoraConfig, TaskType

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import Dataset

    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts) if val_texts else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model="eval_loss" if val_dataset else None,
        report_to="none",
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
    )

    print(f"\nStarting training: {args.epochs} epochs, lr={args.lr}, "
          f"lora_r={args.lora_r}, batch={args.batch_size}x{args.grad_accum}")
    trainer.train()

    # Save LoRA adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapter saved to {adapter_dir}")

    # -----------------------------------------------------------------------
    # Optional: merge into full model
    # -----------------------------------------------------------------------
    if args.merge:
        print("Merging LoRA adapter into base model ...")
        from peft import PeftModel

        # Reload base model in full precision for clean merge
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = merged_model.merge_and_unload()

        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Merged model saved to {merged_dir}")
        print(f"  To serve with vLLM: MODEL={merged_dir} bash scripts/run_vllm_qwen_student.sh")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

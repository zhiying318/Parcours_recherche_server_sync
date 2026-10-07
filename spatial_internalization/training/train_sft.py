#!/usr/bin/env python3
"""Train a fixed-template or verbalized-reasoning LoRA SFT run.

This is ordinary teacher-forced autoregressive cross-entropy.  It does not
import or call any OPSD/self-distillation code and never puts audit geometry in
the user message.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj))|"
    r"(?:linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))|"
    r"(?:mlp\.(?:gate_proj|up_proj|down_proj)))"
)


class CapabilityTrainerMixin:
    def _prepare_inputs(self, inputs):
        # These fields are for local accounting only and are not model inputs.
        inputs.pop("sample_ids", None)
        inputs.pop("target_token_counts", None)
        return super()._prepare_inputs(inputs)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("fixed_template_reasoning", "verbalized_reasoning"),
        required=True,
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B", help="Hub ID for legacy runs")
    parser.add_argument("--model-path", type=Path, default=None, help="complete local base snapshot; preferred for v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "spatial_internalization/datasets/stage1_fixed_split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="train the geometry text as Qwen reasoning_content inside <think>; disabled by default for the comparison run",
    )
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--report-to", default="none", help="none by default; remote logging is deliberately disabled")
    parser.add_argument("--wandb-project", default="spatial-internalization")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_steps == 0 or args.num_train_epochs <= 0:
        raise ValueError("use positive --num-train-epochs or --max-steps=-1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_root = args.data_root.resolve() / args.variant
    train_records = load_jsonl(variant_root / "train.jsonl")
    validation_records = load_jsonl(variant_root / "validation.jsonl")
    if args.model_path is not None:
        model_source = args.model_path.expanduser().resolve()
        from ..core.offline_qwen import resolve_local_model_path

        model_source = resolve_local_model_path(model_source)
        local_files_only = True
    elif args.model_name == "Qwen/Qwen3.5-9B":
        from ..core.offline_qwen import resolve_local_model_path

        model_source = resolve_local_model_path(None)
        local_files_only = True
    else:
        model_source = args.model_name
        local_files_only = False
    processor = AutoProcessor.from_pretrained(str(model_source), local_files_only=local_files_only)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_source),
        local_files_only=local_files_only,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LANGUAGE_LORA_TARGETS,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    collator = __import__("spatial_internalization.core.sft_collator", fromlist=["CapabilitySFTCollator"]).CapabilitySFTCollator(
        processor, REPO_ROOT, max_length=args.max_length, enable_thinking=args.enable_thinking
    )
    report_to = [] if args.report_to.lower() == "none" else [x.strip() for x in args.report_to.split(",") if x.strip()]
    if "wandb" in {name.lower() for name in report_to}:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.run_name:
            os.environ.setdefault("WANDB_NAME", args.run_name)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        tf32=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=report_to,
    )

    class CapabilityTrainer(CapabilityTrainerMixin, Trainer):
        def evaluate(self, *evaluate_args, **evaluate_kwargs):
            was_tracking = collator.track_stats
            collator.track_stats = False
            try:
                return super().evaluate(*evaluate_args, **evaluate_kwargs)
            finally:
                collator.track_stats = was_tracking

    trainer = CapabilityTrainer(
        model=model,
        args=training_args,
        train_dataset=train_records,
        eval_dataset=validation_records,
        data_collator=collator,
        processing_class=processor,
    )
    model.print_trainable_parameters()
    result = trainer.train()
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    config = vars(args).copy()
    config.update({
        "repo_root": str(REPO_ROOT),
        "variant": args.variant,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "train_target_tokens_seen": collator.total_target_tokens,
        "train_examples_seen": collator.total_examples,
        "model_initialized_from": str(model_source),
        "student_enable_thinking": args.enable_thinking,
        "adapter_initialized_from_existing_checkpoint": False,
        "trainable_modules": "language decoder attention/linear-attention/MLP projections matched by LANGUAGE_LORA_TARGETS",
        "external_logging": report_to,
        "train_result": {k: v for k, v in result.metrics.items() if isinstance(v, (str, int, float))},
    })
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

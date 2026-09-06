"""Train Qwen3.5-9B with multimodal on-policy self-distillation."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
)

from self_distillation.vlm_opsd.arguments import parse_bool
from self_distillation.vlm_opsd.collator import VLMOPSDCollator
from self_distillation.vlm_opsd.trainer import VLMOPSDTrainer


REPO_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj))|"
    r"(?:linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))|"
    r"(?:mlp\.(?:gate_proj|up_proj|down_proj)))"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.5-9B",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=REPO_ROOT / "self_distillation/data/train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "self_distillation/output/qwen3_5_9b_opsd_nonthink",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Positive values override --num-train-epochs; defaults to 100 when neither is supplied.")
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--wandb-project", default="opsd-vlm-nonthink-20260905")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume optimizer/model/scheduler/RNG state from a Trainer checkpoint.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--vocabulary-entry-clip", type=float, default=0.05)
    parser.add_argument("--student_thinking", "--student-thinking", type=parse_bool, default=False)
    parser.add_argument("--teacher_thinking", "--teacher-thinking", type=parse_bool, default=False)
    parser.add_argument(
        "--report-to",
        default="wandb",
        help="Comma-separated logging integrations; use 'wandb' after configuring WANDB_API_KEY.",
    )
    parser.add_argument(
        "--debug-diagnostics",
        action="store_true",
        help="Print first-step prompt, completion, logits, gradient, and memory diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=20260827)

    args = parser.parse_args(argv)
    if args.num_train_epochs is not None and args.num_train_epochs <= 0:
        parser.error("--num-train-epochs must be positive")
    if args.max_steps is None:
        args.max_steps = -1 if args.num_train_epochs is not None else 100
    if args.max_steps <= 0 and args.num_train_epochs is None:
        parser.error("Provide --num-train-epochs when --max-steps is non-positive")
    if args.wandb_project == "opsd-vlm":
        parser.error("Use a new --wandb-project for non-thinking experiments")
    return args


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # Keep model and generation metadata in sync with the processor up front,
    # rather than relying on `generate` to reconcile them on its first call.
    for config in (model.config, model.generation_config):
        config.pad_token_id = processor.tokenizer.pad_token_id
        config.bos_token_id = processor.tokenizer.bos_token_id
        config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=LANGUAGE_LORA_TARGETS,
    )
    model = get_peft_model(model, lora_config) # add lora to model
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    report_to = [] if args.report_to.lower() == "none" else [
        name.strip() for name in args.report_to.split(",") if name.strip()
    ]

    train_dataset = load_dataset(
        "json",
        data_files=str(args.train_jsonl),
        split="train",
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs or 1.0,
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=0.1,
        bf16=True,
        tf32=True,
        logging_steps=1,
        save_steps=args.save_steps,
        # Validation is intentionally disabled during training for this run.
        eval_strategy="no",
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=report_to,
        seed=args.seed,
        data_seed=args.seed,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
    )
    trainer = VLMOPSDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VLMOPSDCollator(
            processor, REPO_ROOT,
            student_thinking=args.student_thinking,
            teacher_thinking=args.teacher_thinking,
        ),
        processing_class=processor,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        vocabulary_entry_clip=args.vocabulary_entry_clip,
        debug_diagnostics=args.debug_diagnostics,
    )
    resume_checkpoint = (
        str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint is not None
        else None
    )
    if trainer.is_world_process_zero():
        experiment_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        experiment_config.update(
            world_size=training_args.world_size,
            global_batch_size=(training_args.world_size * args.per_device_batch_size
                               * args.gradient_accumulation_steps),
            train_samples=len(train_dataset),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            lora_rank=64, lora_alpha=128, fixed_teacher=True,
            loss_type="full_vocabulary_forward_kl", samples_per_prompt=1,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "experiment_config.json").write_text(
            json.dumps(experiment_config, indent=2) + "\n"
        )
        if "wandb" in report_to:
            import wandb
            # Explicit ID/project prevent inherited WANDB_* values from resuming an old run.
            wandb.init(project=args.wandb_project, name=args.run_name,
                       id=uuid.uuid4().hex[:12], resume="never", config=experiment_config)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)
        if "wandb" in report_to:
            wandb.finish()


if __name__ == "__main__":
    main()

"""Train Qwen3.5-9B with multimodal on-policy self-distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
)

from self_distillation.archive.vlm_opsd_chunked.collator import VLMOPSDCollator
from self_distillation.archive.vlm_opsd_chunked.trainer import VLMOPSDTrainer


REPO_ROOT = Path(__file__).resolve().parents[3]
LANGUAGE_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj))|"
    r"(?:linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))|"
    r"(?:mlp\.(?:gate_proj|up_proj|down_proj)))"
)


def parse_args() -> argparse.Namespace:
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
        default=REPO_ROOT / "self_distillation/output/qwen3_5_9b_opsd",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-completion-length", type=int, default=512)
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
    parser.add_argument("--loss-chunk-size", type=int, default=16)
    parser.add_argument(
        "--report-to",
        default="none",
        help="Comma-separated logging integrations; use 'wandb' after configuring WANDB_API_KEY.",
    )
    parser.add_argument(
        "--debug-diagnostics",
        action="store_true",
        help="Print first-step prompt, completion, hidden, gradient, and memory diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=20260827)

    parser.add_argument("--max-total-length", type=int, default=65536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=0.1,
        bf16=True,
        tf32=True,
        logging_steps=1,
        save_steps=25,
        # Validation is intentionally disabled during training for this run.
        eval_strategy="no",
        save_total_limit=2,
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
        data_collator=VLMOPSDCollator(processor, REPO_ROOT),
        processing_class=processor,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        vocabulary_entry_clip=args.vocabulary_entry_clip,
        loss_chunk_size=args.loss_chunk_size,
        debug_diagnostics=args.debug_diagnostics,
    )
    resume_checkpoint = (
        str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint is not None
        else None
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

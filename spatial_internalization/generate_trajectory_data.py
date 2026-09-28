#!/usr/bin/env python3
"""Generate SpatialCLI-style trajectory internalization data.

The default ``all`` stage uses one shared Qwen3.5-397B-A17B model through the
configured OpenAI-compatible Chat Completions API for clean evidence
consolidation and the nine train records per family.  The previous local
Qwen3.5-9B backend remains available with ``--provider local`` for debugging.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any

from .data_pipeline import make_options
from .api_qwen import (
    APIRequestError,
    DEFAULT_API_MODEL,
    OpenAICompatibleQwenGenerator,
)
from .geometry import evidence_dict
from .offline_qwen import (
    DEFAULT_MODEL_ID,
    LocalQwenGenerator,
    ContextOverflowError,
    parse_json_object,
    resolve_local_model_path,
)
from .trajectory_pipeline import (
    DEFAULT_FIXED_SPLIT_MANIFEST,
    DEFAULT_SOURCE_CSV,
    DEFAULT_SOURCE_ROOT,
    HISTORICAL_TEACHER_RECORD,
    adapt_evidence,
    apply_rewritten_claims,
    assistant_targets,
    attach_fixed_splits,
    consolidate_record,
    first_version_noise_seeds,
    geometry_for_augmentation,
    load_fixed_split_assignments,
    load_qualified_trajectories,
    write_jsonl,
)
from .trajectory_prompts import (
    consolidation_prompt,
    evidence_rewrite_prompt,
    verbalization_prompt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "spatial_internalization/data_v2"
DEFAULT_CLEAN_EVIDENCE = DEFAULT_OUTPUT_ROOT / "clean_evidence.jsonl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "consolidate", "verbalize"), default="all")
    parser.add_argument(
        "--provider",
        choices=("api", "local"),
        default="api",
        help="offline generator backend; the formal v2 default is the remote API",
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_FIXED_SPLIT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean-evidence-jsonl", type=Path, default=DEFAULT_CLEAN_EVIDENCE)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--augmentations", type=int, default=8)
    parser.add_argument("--position-sigma", type=float, default=0.05)
    parser.add_argument("--angle-sigma-degrees", type=float, default=2.0)
    parser.add_argument("--limit-families", type=int, default=0, help="pilot limit; 0 means all qualified trajectories")
    parser.add_argument("--family-id", action="append", default=[], help="process this image path or trajectory ID; repeatable")
    parser.add_argument("--api-key-env", default=None, help="defaults to OPENAI_API_KEY, then XI_AI_API_KEY")
    parser.add_argument(
        "--api-base-url", "--base-url", dest="api_base_url", default=None,
        help="defaults to OPENAI_BASE_URL or https://api-2.xi-ai.cn/v1",
    )
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-request-retries", type=int, default=5)
    parser.add_argument(
        "--api-context-length",
        type=int,
        default=None,
        help="optional provider context limit for the audit; requests are never truncated",
    )
    parser.add_argument(
        "--consolidator-model-path",
        type=Path,
        default=None,
        help="local Qwen3.5-9B snapshot; default auto-resolves the cached base snapshot",
    )
    parser.add_argument(
        "--verbalizer-model-path",
        type=Path,
        default=None,
        help="local Qwen3.5-9B snapshot; default is the same path as the consolidator",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--context-length", type=int, default=None, help="local provider context override")
    parser.add_argument("--consolidator-max-new-tokens", type=int, default=2048)
    parser.add_argument("--verbalizer-max-new-tokens", type=int, default=2048)
    parser.add_argument("--consolidator-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbalizer-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--consolidator-retries", type=int, default=2)
    parser.add_argument("--verbalizer-retries", type=int, default=2)
    return parser.parse_args(argv)


def _relative(path: Path) -> str:
    # Keep the repository's lexical COMFORT/... symlink path in JSONL instead
    # of turning it into the internal .worktrees/... target.
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def _select_pilot(trajectories: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    selected = trajectories
    if args.family_id:
        wanted = set(args.family_id)
        selected = [
            item for item in selected
            if item["trajectory_id"] in wanted or item["image_path"] in wanted
        ]
    if args.limit_families <= 0 or len(selected) <= args.limit_families:
        return selected
    # A small pilot should cover relations and camera views before adding more
    # examples.  The ordering is deterministic.
    result = []
    seen_relations = set()
    seen_views = set()
    remaining = list(selected)
    while remaining and len(result) < args.limit_families:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                remaining[index]["clean_display"].computed_relation not in seen_relations,
                remaining[index]["camera_view"] not in seen_views,
                -remaining[index]["source_row"],
            ),
        )
        item = remaining.pop(best_index)
        result.append(item)
        seen_relations.add(item["clean_display"].computed_relation)
        seen_views.add(item["camera_view"])
    return sorted(result, key=lambda item: item["source_row"])


def _retry_json_generation(
    generator: Any,
    image_path: Path,
    prompt: str,
    *,
    enable_thinking: bool,
    max_new_tokens: int,
    retries: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error = None
    for attempt in range(1, retries + 2):
        attempt_prompt = prompt if attempt == 1 else (
            prompt + f"\n\nRetry {attempt}: return only valid JSON with no markdown or commentary."
        )
        try:
            text, generation = generator.generate(
                image_path,
                attempt_prompt,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
                attempt=attempt,
            )
            parsed = parse_json_object(text)
            generation["stage"] = label
            generation["raw_output_chars"] = len(text)
            return parsed, generation
        except (ValueError, ContextOverflowError, APIRequestError) as exc:
            last_error = exc
            if isinstance(exc, ContextOverflowError):
                raise
    raise ValueError(f"{label} failed after {retries + 1} attempts: {last_error}")


def _add_generation_totals(
    totals: Counter,
    prefix: str,
    generation: dict[str, Any],
) -> None:
    """Accumulate provider usage without turning unavailable counts into zero."""
    totals[f"{prefix}_calls"] += 1
    for field in ("input_tokens", "output_tokens"):
        value = generation.get(field)
        if value is None:
            totals[f"{prefix}_{field}_unavailable_calls"] += 1
        else:
            totals[f"{prefix}_{field}"] += int(value)
    totals[f"{prefix}_seconds"] += float(generation.get("elapsed_seconds", 0.0))


def _validate_reasoning(
    parsed: dict[str, Any],
    answer_letter: str,
    object_name: str,
    adapted: dict[str, Any],
) -> str:
    reasoning = parsed.get("reasoning_chain")
    final_answer = str(parsed.get("final_answer", "")).strip()
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("verbalizer did not return a non-empty reasoning_chain")
    if final_answer != answer_letter:
        raise ValueError(f"verbalizer final_answer={final_answer!r}, expected {answer_letter!r}")
    lowered = reasoning.lower()
    forbidden = ("tool call", "evidence id", "evidence unit", "consolidation", "verbalization")
    if any(word in lowered for word in forbidden):
        raise ValueError("verbalizer leaked the offline conversion process into the target")
    if object_name.lower() not in lowered:
        raise ValueError("verbalizer omitted the target object")
    if adapted.get("geometry_provenance") != "clean_blender_gt" and not any(
        word in lowered for word in ("estimated", "approximately", "approximate")
    ):
        raise ValueError("perturbed target does not use calibrated estimated/approximately wording")
    if not any(unit["id"] == "G_SPATIAL_CONCLUSION" for unit in adapted["units"]):
        raise ValueError("adapted evidence has no spatial conclusion")
    # A changed vector is represented as a structured canonical unit.  Reject
    # a target that reproduces an entire old vector verbatim for that field.
    old_geometry = adapted.get("clean_geometry_reference", {})
    new_geometry = adapted.get("adapted_geometry", {})
    if adapted.get("changed_evidence_ids"):
        for field in ("person_position", "object_position", "person_forward", "person_right", "relative_vector"):
            old = old_geometry.get(field)
            new = new_geometry.get(field)
            if old != new and old is not None:
                old_text = "[" + ", ".join(f"{float(value):.3f}" for value in old) + "]"
                if old_text in reasoning:
                    raise ValueError(f"verbalizer retained stale {field} vector")
    return reasoning.strip()


def _load_evidence(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["parent_trajectory_id"]] = record
    return records


def _make_augmented_record(
    item: dict[str, Any],
    clean_record: dict[str, Any],
    adapted: dict[str, Any],
    display,
    augmentation_index: int,
    noise_seed: int | None,
    noise_attempts: int,
    reasoning_chain: str,
    seed: int,
    targets: dict[str, str],
    generation: dict[str, Any] | None,
) -> dict[str, Any]:
    question, answer_letter, choices = make_options(
        item["scene"]["object_name"],
        display.computed_relation,
        item["image_path"],
        seed + augmentation_index * 10007,
    )
    family_id = item["trajectory_id"]
    common = {
        "id": f"{family_id.replace(':', '_')}__aug{augmentation_index:02d}",
        "family_id": family_id,
        "parent_trajectory_id": family_id,
        "split": item["split"],
        "object_name": item["scene"]["object_name"],
        "image_path": item["image_path"],
        "scene_gt_path": _relative(item["scene_path"]),
        "scene_group_id": item["scene_group_id"],
        "camera_view": item["camera_view"],
        "question": question,
        "choices": choices,
        "answer_letter": answer_letter,
        "relation": display.computed_relation,
        "augmentation_index": augmentation_index,
        "is_noisy": augmentation_index > 0,
        "clean_gt": clean_record["clean_gt_raw"],
        "clean_display_geometry": clean_record["clean_gt_display"],
        "perturbed_supervision_geometry": evidence_dict(display),
        "parent_raw_trajectory_chars": item["raw_trajectory_chars"],
        "changed_evidence_ids": adapted["changed_evidence_ids"],
        "recomputed_evidence_ids": adapted["recomputed_evidence_ids"],
        "rewritten_claim_ids": adapted["rewritten_claim_ids"],
        "dropped_claims": adapted["dropped_claims"],
        "parent_consolidated_evidence": clean_record["units"],
        "adapted_evidence": adapted["units"],
        "noise_seed": noise_seed,
        "noise_parameters": {
            "position_sigma": 0.05 if augmentation_index else 0.0,
            "angle_sigma_degrees": 2.0 if augmentation_index else 0.0,
        },
        "noise_attempts": noise_attempts,
        "geometry_provenance": adapted["geometry_provenance"],
        "supervision_status": (
            "trajectory-derived, geometry-perturbed supervision"
            if augmentation_index
            else "clean GT-derived supervision"
        ),
    }
    records = []
    for variant, target in targets.items():
        record = dict(common)
        record["variant"] = variant
        record["assistant_target"] = target
        record["generation"] = generation if variant == "trajectory_reasoning" else {"stage": "deterministic"}
        records.append(record)
    return {record["variant"]: record for record in records}


def _target_length_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [len(record["assistant_target"]) for record in records]
    return {
        "count": len(values),
        "min_chars": min(values) if values else 0,
        "mean_chars": statistics.mean(values) if values else 0,
        "max_chars": max(values) if values else 0,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.augmentations != 8:
        raise ValueError("the main experiment is fixed to exactly eight geometric augmentations")
    if args.position_sigma != 0.05 or args.angle_sigma_degrees != 2.0:
        raise ValueError("the main experiment reuses position_sigma=0.05 and angle_sigma_degrees=2.0")
    if args.consolidator_retries < 0 or args.verbalizer_retries < 0:
        raise ValueError("retry counts must be non-negative")

    source_csv = args.source_csv.resolve()
    source_root = args.source_root.expanduser()
    output_root = args.output_root.resolve()
    trajectories, rejected, source_audit = load_qualified_trajectories(source_csv, source_root)
    assignments = load_fixed_split_assignments(args.split_manifest.resolve())
    attach_fixed_splits(trajectories, assignments)
    selected = _select_pilot(trajectories, args)
    if not selected:
        raise ValueError("no qualified trajectories selected")
    noise_seeds = first_version_noise_seeds(
        source_root, args.seed, assignments, args.position_sigma, args.angle_sigma_degrees
    )

    needs_consolidator = args.stage in ("all", "consolidate")
    needs_verbalizer = args.stage in ("all", "verbalize")
    consolidator_path = None
    verbalizer_path = None
    generator = None
    if args.provider == "local":
        consolidator_path = resolve_local_model_path(args.consolidator_model_path, DEFAULT_MODEL_ID) if needs_consolidator else None
        verbalizer_path = resolve_local_model_path(args.verbalizer_model_path, DEFAULT_MODEL_ID) if needs_verbalizer else None
        if args.stage == "all" and consolidator_path != verbalizer_path:
            raise ValueError(
                "--stage all requires consolidator and verbalizer paths to resolve to the same base snapshot; "
                "run the two stages separately when using different paths"
            )
        model_path = consolidator_path or verbalizer_path
        if model_path is not None:
            generator = LocalQwenGenerator(
                model_path,
                device_map=args.device_map,
                attn_implementation=args.attn_implementation,
                context_length=args.context_length,
            )
    else:
        if args.api_request_retries < 0:
            raise ValueError("--api-request-retries must be non-negative")
        generator = OpenAICompatibleQwenGenerator(
            api_key_env=args.api_key_env,
            base_url=args.api_base_url,
            model=args.api_model,
            timeout=args.api_timeout,
            request_retries=args.api_request_retries,
            context_length=args.api_context_length,
        )

    clean_records: dict[str, dict[str, Any]] = {}
    clean_evidence_path = args.clean_evidence_jsonl.resolve()
    if args.stage == "verbalize":
        clean_records = _load_evidence(clean_evidence_path)
        selected = [item for item in selected if item["trajectory_id"] in clean_records]
        if not selected:
            raise ValueError("the clean evidence file contains none of the selected trajectories")

    generation_totals = Counter()
    consolidation_failures = []
    if args.stage in ("all", "consolidate"):
        assert generator is not None
        for item in selected:
            try:
                parsed, generation = _retry_json_generation(
                    generator,
                    REPO_ROOT / item["image_path"],
                    consolidation_prompt(item),
                    enable_thinking=args.consolidator_enable_thinking,
                    max_new_tokens=args.consolidator_max_new_tokens,
                    retries=args.consolidator_retries,
                    label="consolidation",
                )
                record = consolidate_record(item, parsed, generation)
                clean_records[item["trajectory_id"]] = record
                _add_generation_totals(generation_totals, "consolidation", generation)
            except Exception as exc:
                consolidation_failures.append({"parent_trajectory_id": item["trajectory_id"], "reason": str(exc)})
        write_jsonl(clean_evidence_path, clean_records.values())
        if consolidation_failures:
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "audit.json").write_text(
                json.dumps(
                    {
                        "source": source_audit,
                        "rejected": rejected,
                        "consolidation_failures": consolidation_failures,
                        "generation": dict(generation_totals),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"consolidation failed for {len(consolidation_failures)} trajectories; see audit")
        if args.stage == "consolidate":
            print(json.dumps({"clean_evidence": str(clean_evidence_path), "count": len(clean_records)}, indent=2))
            return

    variants = {"answer_only": [], "template_reasoning": [], "trajectory_reasoning": []}
    family_failures = []
    if args.stage in ("all", "verbalize"):
        assert generator is not None
        for item in selected:
            clean_record = clean_records.get(item["trajectory_id"])
            if clean_record is None:
                family_failures.append({"parent_trajectory_id": item["trajectory_id"], "reason": "missing_clean_consolidation"})
                continue
            indexes = range(9) if item["split"] == "train" else range(1)
            family_records = {name: [] for name in variants}
            family_failed = None
            for augmentation_index in indexes:
                noise_seed = None if augmentation_index == 0 else noise_seeds.get((item["image_path"], augmentation_index))
                if augmentation_index > 0 and noise_seed is None:
                    family_failed = "missing_first_version_noise_seed"
                    break
                try:
                    display, attempts = geometry_for_augmentation(
                        item,
                        augmentation_index,
                        noise_seed,
                        args.position_sigma,
                        args.angle_sigma_degrees,
                    )
                    adapted = adapt_evidence(
                        clean_record,
                        display,
                        item["scene"]["object_name"],
                        augmentation_index,
                    )
                    rewrite_generation = None
                    if adapted["claims_to_rewrite"]:
                        rewritten, rewrite_generation = _retry_json_generation(
                            generator,
                            REPO_ROOT / item["image_path"],
                            evidence_rewrite_prompt(adapted),
                            enable_thinking=args.consolidator_enable_thinking,
                            max_new_tokens=args.consolidator_max_new_tokens,
                            retries=args.consolidator_retries,
                            label="evidence_rewrite",
                        )
                        adapted = apply_rewritten_claims(adapted, rewritten)
                        _add_generation_totals(
                            generation_totals, "evidence_rewrite", rewrite_generation
                        )
                    question, answer_letter, choices = make_options(
                        item["scene"]["object_name"],
                        display.computed_relation,
                        item["image_path"],
                        args.seed + augmentation_index * 10007,
                    )
                    prompt = verbalization_prompt(item, adapted, answer_letter, choices)
                    parsed, generation = _retry_json_generation(
                        generator,
                        REPO_ROOT / item["image_path"],
                        prompt,
                        enable_thinking=args.verbalizer_enable_thinking,
                        max_new_tokens=args.verbalizer_max_new_tokens,
                        retries=args.verbalizer_retries,
                        label="verbalization",
                    )
                    reasoning = _validate_reasoning(
                        parsed, answer_letter, item["scene"]["object_name"], adapted
                    )
                    targets = assistant_targets(
                        item["scene"]["object_name"], display, reasoning, answer_letter
                    )
                    built = _make_augmented_record(
                        item,
                        clean_record,
                        adapted,
                        display,
                        augmentation_index,
                        noise_seed,
                        attempts,
                        reasoning,
                        args.seed,
                        targets,
                        {
                            "verbalization": generation,
                            "evidence_rewrite": rewrite_generation,
                        },
                    )
                    for variant in variants:
                        family_records[variant].append(built[variant])
                    _add_generation_totals(generation_totals, "verbalization", generation)
                except Exception as exc:
                    family_failed = str(exc)
                    break
            if family_failed:
                family_failures.append({
                    "parent_trajectory_id": item["trajectory_id"],
                    "split": item["split"],
                    "reason": family_failed,
                    "family_size_required": 9 if item["split"] == "train" else 1,
                })
                continue
            for variant in variants:
                variants[variant].extend(family_records[variant])

    data_root = output_root / "data"
    for variant, records in variants.items():
        for split in ("train", "validation", "test"):
            write_jsonl(
                data_root / variant / f"{split}.jsonl",
                [record for record in records if record["split"] == split],
            )

    all_c_records = variants["trajectory_reasoning"]
    assert generator is not None
    api_metadata = generator.metadata if args.provider == "api" else {}
    raw_trajectory_tokens = [generator.count_text_tokens(item["row"]["model_answer"]) for item in selected]
    clean_evidence_tokens = [
        generator.count_text_tokens(json.dumps(record["units"], ensure_ascii=False))
        for record in clean_records.values()
    ]
    clean_c_records = [record for record in all_c_records if record["augmentation_index"] == 0]
    augmented_c_records = [record for record in all_c_records if record["augmentation_index"] > 0]
    manifest = {
        "experiment": "spatial_internalization_v2_trajectory_guided",
        "paper_alignment": {
            "consolidation": "SpatialCLI Sec. 3.3 / Appendix G.2 adapted to one retained CoT + Blender GT unit",
            "verbalization": "SpatialCLI Sec. 3.3 / Appendix G.3",
            "training": "ordinary mean autoregressive NLL/SFT; no distillation",
        },
        "source": source_audit,
        "historical_teacher_model": HISTORICAL_TEACHER_RECORD,
        "fixed_split_manifest": _relative(args.split_manifest),
        "fixed_split_assignments_used": assignments,
        "selected_family_count": len(selected),
        "complete_family_count": len({record["family_id"] for record in all_c_records}),
        "complete_family_counts_by_split": {
            split: len({record["family_id"] for record in all_c_records if record["split"] == split})
            for split in ("train", "validation", "test")
        },
        "record_counts": {
            variant: {
                split: sum(record["split"] == split for record in records)
                for split in ("train", "validation", "test")
            }
            for variant, records in variants.items()
        },
        "augmentation": {
            "train_records_per_family": 9,
            "validation_test_records_per_family": 1,
            "position_sigma": args.position_sigma,
            "angle_sigma_degrees": args.angle_sigma_degrees,
            "label_preserving_rejection": True,
            "image_is_unchanged": True,
            "perturbed_geometry_is_not_clean_gt": True,
            "noise_seed_rule": "reused spatial_internalization.data_pipeline first-version expression",
        },
        "generation": {
            "provider": args.provider,
            "consolidator_model_path": str(consolidator_path) if consolidator_path else None,
            "verbalizer_model_path": str(verbalizer_path) if verbalizer_path else None,
            "api_base_url": api_metadata.get("api_base_url"),
            "api_model": args.api_model if args.provider == "api" else None,
            "api_key_env": api_metadata.get("api_key_env"),
            "api_timeout": args.api_timeout if args.provider == "api" else None,
            "api_request_retries": args.api_request_retries if args.provider == "api" else None,
            "api_context_length": args.api_context_length if args.provider == "api" else None,
            "sdk_request_path": api_metadata.get("sdk_request_path"),
            "shared_model_instance_in_all_stage": args.stage == "all",
            "consolidator_enable_thinking": args.consolidator_enable_thinking,
            "verbalizer_enable_thinking": args.verbalizer_enable_thinking,
            "consolidator_max_new_tokens": args.consolidator_max_new_tokens,
            "verbalizer_max_new_tokens": args.verbalizer_max_new_tokens,
            "context_length": generator.context_length if generator else None,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "token_and_time_totals": dict(generation_totals),
        },
        "length_distributions": {
            "raw_trajectory_chars": _distribution([item["raw_trajectory_chars"] for item in selected]),
            "raw_trajectory_tokens": _distribution(raw_trajectory_tokens),
            "clean_evidence_chars": _distribution([len(json.dumps(record["units"], ensure_ascii=False)) for record in clean_records.values()]),
            "clean_evidence_tokens": _distribution(clean_evidence_tokens),
            "trajectory_target_chars": _target_length_stats(all_c_records),
            "trajectory_clean_target_chars": _target_length_stats(clean_c_records),
            "trajectory_augmented_target_chars": _target_length_stats(augmented_c_records),
            "trajectory_target_tokens": _distribution([generator.count_text_tokens(record["assistant_target"]) for record in all_c_records]),
            "trajectory_clean_target_tokens": _distribution([generator.count_text_tokens(record["assistant_target"]) for record in clean_c_records]),
            "trajectory_augmented_target_tokens": _distribution([generator.count_text_tokens(record["assistant_target"]) for record in augmented_c_records]),
            "template_target_chars": _target_length_stats(variants["template_reasoning"]),
            "template_target_tokens": _distribution([generator.count_text_tokens(record["assistant_target"]) for record in variants["template_reasoning"]]),
            "answer_target_chars": _target_length_stats(variants["answer_only"]),
            "answer_target_tokens": _distribution([generator.count_text_tokens(record["assistant_target"]) for record in variants["answer_only"]]),
        },
        "rejected_source_rows": rejected,
        "consolidation_failures": consolidation_failures,
        "family_failures": family_failures,
        "pilot": {"limit_families": args.limit_families, "family_ids": args.family_id},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (output_root / "audit.json").write_text(json.dumps({"source": source_audit, "rejected": rejected, "family_failures": family_failures}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


def _distribution(values: list[int | None]) -> dict[str, Any]:
    available = [int(value) for value in values if value is not None]
    return {
        "count": len(values),
        "available_count": len(available),
        "min": min(available) if available else None,
        "mean": statistics.mean(available) if available else None,
        "max": max(available) if available else None,
    }


if __name__ == "__main__":
    main()

"""Prompts for the two local Qwen stages.

The prompts follow SpatialCLI's separation: consolidation receives no answer
key, while global verbalization receives only the already consolidated,
adapted evidence and the final answer needed to order it.
"""

from __future__ import annotations

import json
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def consolidation_prompt(item: dict[str, Any]) -> str:
    row = item["row"]
    clean = item["clean_display"]
    geometry = {
        "camera_coordinate_system": "+X is image-right, +Y is downward, +Z is forward into the scene",
        "person_position": list(clean.person_position),
        "object_position": list(clean.object_position),
        "person_forward_axis": list(clean.person_forward),
        "person_right_axis": list(clean.person_right),
        "person_to_object_displacement": list(clean.relative_vector),
        "forward_projection": clean.forward_projection,
        "right_projection": clean.right_projection,
        #"computed_relation": clean.computed_relation,
        "provenance": "clean Blender GT, normalized and quantized with the first-version rule",
    }
    return f"""You are performing turn-wise evidence consolidation for capability-internalization data.

This is one successful visual reasoning trajectory. Convert the original model reasoning and the clean geometry GT into a structured list of evidence claims. The answer key is deliberately withheld at this stage. Do not solve the multiple-choice question and do not output a letter.

Rules:
1. Every claim must be traceable to the clean geometry GT below or to an explicit statement in the original trajectory. Do not treat plans, guesses, self-corrections that are not supported, or option speculation as observations.
2. Preserve the geometry fields and all explicit visual or spatial facts supported by the trajectory or geometry, but do not copy the entire repetitive chain.
3. Every claim needs an ID, a short readable text, a list of depends_on IDs, and fields. Use G_* IDs for the supplied geometry fields and COT_* IDs for claims extracted from the trajectory. A claim that depends on a position, axis, displacement, projection, or spatial conclusion must name those G_* IDs.
4. Do not discard an explicit supported fact merely because it is unnecessary for selecting the final answer.
5. Do not add entities, values, or relations that are absent from the GT or trajectory.
6. Do not mention this prompt in any claim.

Return JSON only, with this exact outer shape:
{{"evidence":[{{"id":"COT_001","text":"...","depends_on":["G_..."],"fields":["..."]}}]}}

<image_task>
{row['mcq_prompt']}
</image_task>

<clean_geometry_gt>
{_json(geometry)}
</clean_geometry_gt>

<original_trajectory_reasoning>
{row['model_answer']}
</original_trajectory_reasoning>
"""


def verbalization_prompt(
    item: dict[str, Any],
    adapted_evidence: dict[str, Any],
    answer_letter: str,
    choices: dict[str, str],
) -> str:
    return f"""You are preparing a compact tool-free reasoning target for vision-language SFT.

Use only the adapted evidence below. Write direct visual-spatial reasoning that explains the final multiple-choice answer. Do not mention tools, calls, trajectories, evidence IDs, consolidation, verbalization, prompts, confidence, or this conversion process. Do not use any old geometry values that are absent from the adapted evidence. Use calibrated wording such as estimated or approximately because the geometry is perceptual supervision; for augmentation records it is explicitly perturbed supervision rather than a new teacher trajectory.

The reasoning should be complete and detailed, preserve the camera convention, state the relevant positions/axes or displacement and projections, explain the dominant component and sign, and support the provided answer. Do not invent any entity, value, attribute, or relation. The final answer must be exactly the supplied letter.

Return JSON only:
{{"reasoning_chain":"...","final_answer":"{answer_letter}"}}

<task_question>
Where is the {item['scene']['object_name']} in the perspective of the person?
</task_question>

<choices>
{_json(choices)}
</choices>

<adapted_evidence>
{_json(adapted_evidence['units'])}
</adapted_evidence>

<correct_final_answer>
{answer_letter}
</correct_final_answer>
"""


def evidence_rewrite_prompt(adapted_evidence: dict[str, Any]) -> str:
    rewrite_ids = set(adapted_evidence["claims_to_rewrite"])
    claims = [
        {"id": unit["id"], "text": unit["text"]}
        for unit in adapted_evidence["units"]
        if unit["id"] in rewrite_ids
    ]
    geometry = [
        unit for unit in adapted_evidence["units"] if unit["id"].startswith("G_")
    ]
    return f"""Rewrite the listed reasoning claims so that they agree with the updated geometry evidence.

Preserve each claim's reasoning role and level of detail. Replace stale geometry values, comparisons, signs, and spatial conclusions. Use only the updated geometry evidence. Do not mention rewriting, perturbation, evidence IDs, prompts, or this process.

Return every listed claim exactly once, with the same ID, as JSON only:
{{"rewritten_claims":[{{"id":"COT_001","text":"..."}}]}}

<updated_geometry_evidence>
{_json(geometry)}
</updated_geometry_evidence>

<claims_to_rewrite>
{_json(claims)}
</claims_to_rewrite>
"""

"""Build student-only and geometry-privileged Qwen3.5-VL batches."""

from __future__ import annotations

from pathlib import Path


TEACHER_CONTEXT = """\
Here is a reference solution.
=== Reference Solution Begin ===
{teacher_geometry}
=== Reference Solution End ===
After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:
"""

class VLMOPSDCollator:
    """Encode the same image under student and privileged teacher prompts.

    Qwen3.5-VL's processor expands image placeholders to the correct patch-token
    count and creates the multimodal RoPE metadata. A tokenizer-only collator
    cannot produce those tensors.
    """

    def __init__(self, processor, repo_root: str | Path):
        self.processor = processor
        self.repo_root = Path(repo_root).resolve()
        self.processor.tokenizer.padding_side = "left"

    @staticmethod
    def _conversation(image_path: str, text: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def _encode(self, conversations: list[list[dict]]):
        return self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_resize": False, "padding": True},
            enable_thinking=True,
        )

    def __call__(self, features: list[dict]):
        image_paths = [str(self.repo_root / feature["image_path"]) for feature in features]
        student_conversations = [
            self._conversation(image_path, feature["problem"])
            for image_path, feature in zip(image_paths, features)
        ]
        teacher_conversations = [
            self._conversation(
                image_path,
                feature["problem"]
                + "\n\n"
                + TEACHER_CONTEXT.format(
                    teacher_geometry=feature["teacher_geometry"]
                ),
            )
            for image_path, feature in zip(image_paths, features)
        ]

        student = self._encode(student_conversations)
        teacher = self._encode(teacher_conversations)
        batch = {f"student_{key}": value for key, value in student.items()}
        batch.update({f"teacher_{key}": value for key, value in teacher.items()})
        batch["sample_ids"] = [feature["id"] for feature in features]
        return batch

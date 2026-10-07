import unittest
from pathlib import Path

import torch

from spatial_internalization.core.sft_collator import CapabilitySFTCollator


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, conversations, **kwargs):
        full = conversations[0][-1]["role"] == "assistant"
        rows = []
        for conversation in conversations:
            length = 5 if conversation[-1]["role"] == "assistant" else 3
            rows.append(list(range(1, length + 1)))
        width = max(len(row) for row in rows)
        ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        mask = (ids != 0).long()
        return {"input_ids": ids, "attention_mask": mask, "pixel_values": torch.ones(len(rows), 1)}


class SerializationTests(unittest.TestCase):
    def test_only_assistant_target_has_loss(self):
        collator = CapabilitySFTCollator(FakeProcessor(), Path.cwd())
        result = collator([{
            "id": "x",
            "image_path": "some.png",
            "question": "Where is the ball?",
            "choices": {"A": "in front", "B": "behind", "C": "left", "D": "right"},
            "assistant_target": "A",
        }])
        self.assertEqual(result["labels"].tolist(), [[-100, -100, -100, 4, 5]])
        self.assertEqual(result["target_token_counts"], [2])
        self.assertNotIn("sample_ids", {})


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import nullcontext
import io

from self_distillation.vlm_opsd.collator import VLMOPSDCollator
from self_distillation.vlm_opsd.train import parse_args

from self_distillation.vlm_opsd.trainer import VLMOPSDTrainer, completion_mask, forward_kl


class VLMOPSDTest(unittest.TestCase):
    def test_completion_mask_includes_only_first_eos(self):
        tokens = torch.tensor(
            [
                [4, 5, 9, 0, 0],
                [4, 8, 5, 9, 0],
                [4, 5, 6, 7, 8],
            ]
        )
        actual = completion_mask(tokens, [8, 9])
        expected = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_forward_kl_is_zero_for_equal_distributions(self):
        hidden = torch.tensor(
            [[[1.0, 2.0], [3.0, 2.0]]], requires_grad=True
        )
        loss = forward_kl(
            student_logits=hidden,
            teacher_logits=hidden.detach().clone(),
            mask=torch.ones(1, 2),
            temperature=1.0,
            vocabulary_entry_clip=None,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_forward_kl_ignores_masked_completion_tokens(self):
        student = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
        masked = forward_kl(
            student_logits=student,
            teacher_logits=teacher,
            mask=torch.tensor([[0, 1]]),
            temperature=1.0,
            vocabulary_entry_clip=None,
        )
        self.assertAlmostEqual(masked.item(), 0.0, places=6)

    def test_forward_kl_backpropagates_through_student(self):
        student = torch.randn(1, 5, 3, requires_grad=True)
        teacher = torch.randn(1, 5, 3)
        loss = forward_kl(
            student_logits=student,
            teacher_logits=teacher,
            mask=torch.ones(1, 5),
            temperature=1.0,
            vocabulary_entry_clip=None,
        )
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.norm().item(), 0.0)

    def test_thinking_cli_accepts_explicit_false_and_independent_modes(self):
        args = parse_args(["--student_thinking", "False", "--teacher_thinking", "True"])
        self.assertIs(args.student_thinking, False)
        self.assertIs(args.teacher_thinking, True)
        self.assertIs(parse_args([]).teacher_thinking, False)
        self.assertIs(parse_args(["--student-thinking", "false"]).student_thinking, False)
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            parse_args(["--student_thinking", "typo"])

    def test_collator_passes_independent_thinking_flags_and_images(self):
        for student in (False, True):
            for teacher in (False, True):
                processor = SimpleNamespace(
                    tokenizer=SimpleNamespace(padding_side="right"),
                    apply_chat_template=Mock(return_value={"input_ids": torch.ones(1, 2)}),
                )
                collator = VLMOPSDCollator(
                    processor, "/tmp", student_thinking=student, teacher_thinking=teacher
                )
                collator([{"id": "sample", "image_path": "image.png", "problem": "Where?",
                           "teacher_geometry": "position=(1,2,3)"}])
                calls = processor.apply_chat_template.call_args_list
                self.assertEqual([c.kwargs["enable_thinking"] for c in calls], [student, teacher])
                student_content = calls[0].args[0][0][0]["content"]
                teacher_content = calls[1].args[0][0][0]["content"]
                self.assertEqual(student_content[0], teacher_content[0])
                self.assertNotIn("position=", student_content[1]["text"])
                self.assertIn("position=", teacher_content[1]["text"])

    def test_kl_matches_reference_clipping_and_stops_teacher_gradient(self):
        student = torch.tensor([[[2., 0., -1.], [1., 3., 0.]]], requires_grad=True)
        teacher = torch.tensor([[[0., 2., 1.], [3., 0., 1.]]], requires_grad=True)
        mask = torch.tensor([[1, 0]])
        temperature, clip = 0.7, 0.05
        loss = forward_kl(student, teacher, mask, temperature, clip)
        p = torch.softmax(teacher.detach()[0, 0] / temperature, -1)
        log_q = torch.log_softmax(student[0, 0] / temperature, -1)
        expected = (p * (p.log() - log_q)).clamp(max=clip).sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[0, 1].abs().sum().item(), 0.)

    def test_compute_loss_aligns_shared_completion_and_preserves_multimodal_inputs(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(10, 10))
                self.config = SimpleNamespace(use_cache=False)
                self.calls = []

            def generate(self, input_ids, **kwargs):
                self.rollout_kwargs = kwargs
                return torch.cat([input_ids, torch.tensor([[6, 9, 0], [7, 8, 9]])], dim=1)

            def forward(self, input_ids, **kwargs):
                self.calls.append((input_ids.clone(), kwargs, torch.is_grad_enabled()))
                return SimpleNamespace(logits=self.weight[input_ids])

        model = TinyModel()
        trainer = object.__new__(VLMOPSDTrainer)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
        trainer.debug_diagnostics = False
        trainer.temperature = 1.0
        trainer.vocabulary_entry_clip = None
        trainer.generation_config = SimpleNamespace(eos_token_id=9)
        trainer._fixed_teacher_context = Mock(side_effect=lambda m: nullcontext())
        inputs = {}
        for prefix, ids in (("student", [[0, 1, 2], [1, 2, 3]]),
                            ("teacher", [[0, 1, 4, 5], [1, 4, 5, 6]])):
            ids = torch.tensor(ids)
            fields = {"input_ids": ids, "attention_mask": ids.ne(0).long(),
                      "pixel_values": torch.ones(2, 4), "image_grid_thw": torch.ones(2, 3),
                      "mm_token_type_ids": torch.ones_like(ids)}
            inputs.update({prefix + "_" + k: v for k, v in fields.items()})
        loss = trainer.compute_loss(model, inputs)
        teacher_ids, teacher_kwargs, teacher_grad = model.calls[0]
        student_ids, student_kwargs, student_grad = model.calls[1]
        self.assertFalse(teacher_grad)
        self.assertTrue(student_grad)
        trainer._fixed_teacher_context.assert_called_once_with(model)
        torch.testing.assert_close(student_ids[:, 3:], teacher_ids[:, 4:])
        self.assertIs(student_kwargs["pixel_values"], inputs["student_pixel_values"])
        self.assertIs(teacher_kwargs["image_grid_thw"], inputs["teacher_image_grid_thw"])
        torch.testing.assert_close(student_kwargs["attention_mask"][:, 3:], torch.tensor([[1, 1, 0], [1, 1, 1]]))
        self.assertEqual(student_kwargs["mm_token_type_ids"][:, 3:].sum().item(), 0)
        expected = forward_kl(model.weight[student_ids][:, 2:-1],
                              model.weight[teacher_ids][:, 3:-1],
                              torch.tensor([[1, 1, 0], [1, 1, 1]]), 1., None)
        torch.testing.assert_close(loss, expected)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())


if __name__ == "__main__":
    unittest.main()

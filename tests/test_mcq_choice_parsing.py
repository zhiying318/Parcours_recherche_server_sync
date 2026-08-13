import unittest

from spatial_eval.prompts.MCQ import _normalize_choice_thinking


class ThinkingChoiceParsingTests(unittest.TestCase):
    def test_standalone_terminal_choice(self):
        self.assertEqual(_normalize_choice_thinking("<think>A or B?</think>\nD"), "D")

    def test_tagged_terminal_choice(self):
        response = "<think>Reasoning.</think>\n<answer>B</answer>"
        self.assertEqual(_normalize_choice_thinking(response), "B")

    def test_explicit_terminal_choice(self):
        response = "<think>Reasoning.</think>\nThe final answer is C."
        self.assertEqual(_normalize_choice_thinking(response), "C")

    def test_full_option_terminal_choice(self):
        response = "<think>Reasoning.</think>\nD. The object is behind the person."
        self.assertEqual(_normalize_choice_thinking(response), "D")

    def test_article_is_not_an_answer(self):
        response = "<think>Reasoning.</think>\nA person is visible in the image."
        self.assertEqual(_normalize_choice_thinking(response), "")

    def test_truncated_thinking_is_incomplete(self):
        response = "The reasoning is unfinished, so perhaps option A"
        self.assertEqual(_normalize_choice_thinking(response), "")


if __name__ == "__main__":
    unittest.main()

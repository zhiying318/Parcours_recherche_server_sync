# spatial_eval/backends/base.py
from typing import List

class VLMBackend:
    def ask(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError

    def ask_multi(self, image_paths: List[str], prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError
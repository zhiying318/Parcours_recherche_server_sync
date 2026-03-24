# spatial_eval/backends/base.py
class VLMBackend:
    def ask(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError
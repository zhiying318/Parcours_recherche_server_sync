# Archived thinking / chunked implementation

Snapshot of the previous implementation: both prompts enable thinking; the trainer
captures lm_head inputs and computes checkpointed KL in token chunks.
The main `self_distillation.vlm_opsd.train` entry point does not import this code.
For historical reproduction only, use
`python -m self_distillation.archive.vlm_opsd_chunked.train` with the old arguments.
Existing datasets and model checkpoints have not been moved or changed.

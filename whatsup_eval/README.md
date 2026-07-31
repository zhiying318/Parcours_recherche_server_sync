# WhatsUp MCQ Evaluation

This folder evaluates the validated WhatsUp human-centered images with the same
single-image long MCQ format used in COMFORT.

The canonical 563-image manifest is
`whatsup_image_validation/valide_image_paths.json`. Run the existing evaluation from the
project root with:

```bash
./whatsup_eval/run.sh
```

## Validation

All validation code and artifacts are under `whatsup_image_validation/`:

- `get_valid_image_paths.py`: reproducible two-round Qwen3-VL validator
- `validation_by_mLLM.ipynb`: original development notebook
- `validation_qwen3vl.csv`: first-round answers for 630 candidates
- `validation_qwen3vl_2nd.csv`: second-round answers for 567 candidates
- `valide_image_paths.json`: final 563-image manifest

The first round records four questions: one human, object next to the human,
object present, and object recognisable. To preserve the rule that produced the
current dataset, an image enters the second round when **at least one** of the
three object checks is `yes`; the one-human answer is recorded but is not used
as a filter. The second round keeps images only when the human is complete and
there is margin above the head.

Regenerating the CSV files and manifest invokes the local Qwen3-VL model and
overwrites the current validation artifacts:

```bash
CUDA_VISIBLE_DEVICES=0 \
python whatsup_eval/whatsup_image_validation/get_valid_image_paths.py
```

Normal evaluation does not need to rerun validation.

The evaluator computes ground truth from filenames:

- `*_left_of_*_FACE-CAMERA` -> `right`
- `*_right_of_*_FACE-CAMERA` -> `left`
- `*_left_of_*_FACE-LEFT` -> `front`
- `*_right_of_*_FACE-LEFT` -> `behind`
- `*_left_of_*_FACE-RIGHT` -> `behind`
- `*_right_of_*_FACE-RIGHT` -> `front`

Summarize results:

```bash
python analyse_results.py results/*.csv
```

## GPT-5 via the validated OpenAI-compatible endpoint

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL_ID` in the
environment. The scripts never store or print the API key.

Instruct mode uses Chat Completions:

```bash
export OPENAI_MODEL_ID="gpt-5"
export OPENAI_RUN_MODE="instruct"
export OPENAI_API_MODE="chat_completions"
./whatsup_eval/run_openai_smoke.sh 0
./whatsup_eval/run_openai.sh
```

Thinking mode uses the Responses API:

```bash
export OPENAI_MODEL_ID="gpt-5"
export OPENAI_RUN_MODE="thinking"
export OPENAI_API_MODE="responses"
export OPENAI_REASONING_EFFORT="high"
export OPENAI_REASONING_SUMMARY="auto"
./whatsup_eval/run_openai_smoke.sh 0
./whatsup_eval/run_openai.sh
```

Smoke tests select one image and default to zero SDK retries, so each invocation
attempts exactly one paid request. Full runs resume compatible CSV files,
flush every successful row, and write failures and reasoning summaries to
separate JSONL sidecars.

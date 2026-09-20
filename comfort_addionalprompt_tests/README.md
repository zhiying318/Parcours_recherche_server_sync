# Single-image viewpoint prompt ablations

These experiments use the existing `spatial_eval` single-image MCQ evaluator,
option randomizer, seed, scoring, and model backends.  The only new evaluator
input is an optional JSON mapping from image path to information inserted into
that image's prompt.

All experiments use the same 144 `0.png` renders from
`COMFORT/data/comfort_human_car_geometry_gt/comfort_human_car`.

| Directory | Extra information in the prompt |
|---|---|
| `test00_baseline` | none |
| `test01_camera_side` | camera's categorical location relative to the person |
| `test02_camera_coordinate_system` | camera-side information plus camera-coordinate convention |
| `test03_object_camera_xyz` | object's visible-center coordinates in the OpenCV camera frame |
| `test04_person_object_camera_xyz` | person and object visible-center camera coordinates |
| `test05_person_forward_axis` | all previous information plus the person's forward axis in the camera frame |
| `test06_camera_coordinate_system_person_forward_axis` | camera-origin convention plus the person's forward axis only, placed before the question |
| `test07_camera_geometry_before_question` | camera-origin person/object coordinates and person forward axis, placed before the question |

Each test contains all information from the preceding test and adds one new
piece of information. No world coordinate system is used.

Generate/refresh all JSON data from the Blender GT:

```bash
python comfort_addionalprompt_tests/generate_data.py
```

## Docker execution

The non-API model evaluations should use the environment inside the Docker
container. The host Python/conda environment is not required. The Docker image,
model cache, and Python packages are configured by
`run_docker_gpu_COMFORT.sh`.

`HF_ENDPOINT` is injected into the container automatically and defaults to
`https://huggingface.co`. Set it explicitly only when using another mirror.

Run `test07` independently for each model family:

```bash
QWEN_GPU=2 \
bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh \
qwen35 test07_camera_geometry_before_question

GEMMA_GPU=3 \
bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh \
gemma test07_camera_geometry_before_question

INTERNVL_GPU=0 \
bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh \
internvl test07_camera_geometry_before_question
```

The Qwen, Gemma, and InternVL runners each execute their non-thinking and
thinking evaluations separately. Results are written under the test directory
in `results_preciseprompt/`. Existing compatible CSVs are resumed.

The commands above expose all GPUs to Docker and select the model GPU with
`CUDA_VISIBLE_DEVICES`. To expose only one physical GPU to the container, use
container GPU `0`, for example:

```bash
COMFORT_ADD_PROMPT_GPU=2 INTERNVL_GPU=0 \
bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh \
internvl test07_camera_geometry_before_question
```

The API-based GPT-5 runner is separate and is not launched through this Docker
script.

InternVL and Gemma4 downloads use the existing HTTP(S) proxy and disable Xet by default
(`HF_HUB_DISABLE_XET=1`). Before evaluation, `recover_hf_weights.py` downloads
missing weight shards in 32 MiB HTTP ranges with four workers, bounded retries,
and SHA256 verification. Completed ranges survive restarts; complete cached
shards are reused. This avoids long weight-download connections that stall or
close prematurely through the server proxy. No host Python environment is used.
The downloader supports both sharded checkpoints and a single `model.safetensors`,
and uses the existing Hugging Face credentials for authenticated requests.
For test07 on physical GPU 4:

```bash
COMFORT_ADD_PROMPT_GPU=4 INTERNVL_GPU=0 \
bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh \
internvl test07_camera_geometry_before_question
```

For Gemma4 on physical GPU 1, use `COMFORT_ADD_PROMPT_GPU=1 GEMMA_GPU=0`
with the same command and replace `internvl` with `gemma`.

## Direct host execution

Run one model family directly from the repository root only when the host
environment contains the required model dependencies:

```bash
QWEN_GPU=0 bash comfort_addionalprompt_tests/test01_camera_side/run_qwen35.sh
GEMMA_GPU=1 bash comfort_addionalprompt_tests/test01_camera_side/run_gemma.sh
```

The model-specific runners are `run_experiment_qwen35.sh` and
`run_experiment_gemma.sh`. Both use the same shared setup, prompt inputs,
randomization, output directory, and resume logic.

The Qwen, Gemma, and InternVL runners evaluate non-thinking and then thinking.
Existing compatible CSVs are resumed, and `PYTHONHASHSEED=0` keeps the
hash-based option order identical across model processes.

The complete single-image MCQ prompt has this fixed structure (the option
order is randomized by the existing code):

```text
Where is the {object} in the perspective of the person?
{test-specific additional information, if any}
Choose ONE option and respond with ONLY the letter.
A. From the person's perspective, the {object} is in front of them.
B. From the person's perspective, the {object} is behind them.
C. From the person's perspective, the {object} is on their left.
D. From the person's perspective, the {object} is on their right.
```

For `test00`, the second line is absent. The additional-information lines are:

```text
test01: The viewpoint from which this image was captured is located {in front of / behind / to the left of / to the right of} the depicted person.
test02: test01 + In the camera coordinate system, +X points to the image's right, +Y points downward, and +Z points forward into the scene.
test03: test02 + The coordinates of the {object} from the camera's perspective are {...}.
test04: test03 + The coordinates of the person from the camera's perspective are {...}.
test05: test04 + In the camera coordinate system, the person's forward direction is {...}.
test06: Consider that the picture was taken from the origin [0.000, 0.000, 0.000] of a camera coordinate system, where +X points to the image's right, +Y points downward, and +Z points forward into the scene. The person in the image is looking in the direction of {...} (unit vector).
test07: Consider that the picture was taken from the origin [0.000, 0.000, 0.000] of a camera coordinate system, where +X points to the image's right, +Y points downward, and +Z points forward into the scene. The person in the image is located at {...} and is looking in the direction of {...} (unit vector). The {object} is located at {...}.
```

Unlike tests 01–05, test06 is not cumulative. It deliberately excludes the
camera-side description and the object/person coordinates, isolating whether
the camera-coordinate convention plus the person's orientation is sufficient.
Tests 06–07 place their per-image geometry before the question. Test07 is also
non-cumulative; its positions and direction are expressed in the camera frame,
not in Blender's person-centered world frame.

## Thinking-answer parsing

Thinking responses are scored only from the answer section after `</think>`.
The parser accepts a standalone terminal letter, an explicit statement such as
`The final answer is D`, a full option line beginning with `D.`, or an
`<answer>D</answer>` tag. It does not uppercase and search the prose because an
English article such as `a` would then be mistaken for option A.

If generation reaches its token limit before `</think>`, `pred_letter` is left
empty and the row is counted as an incomplete generation, not guessed from a
partial chain of thought. Existing result files can be rescored without model
inference:

```bash
python comfort_addionalprompt_tests/reparse_thinking_results.py \
  path/to/mcq_long_qwen3_5vl_thinking.csv
```

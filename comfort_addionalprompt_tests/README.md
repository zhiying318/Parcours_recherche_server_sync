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
| `test06_camera_coordinate_system_person_forward_axis` | camera-coordinate convention plus the person's forward axis only |

Each test contains all information from the preceding test and adds one new
piece of information. No world coordinate system is used.

Generate/refresh all JSON data from the Blender GT:

```bash
python comfort_addionalprompt_tests/generate_data.py
```

Run one model family directly from the repository root:

```bash
QWEN_GPU=0 bash comfort_addionalprompt_tests/test01_camera_side/run_qwen35.sh
GEMMA_GPU=1 bash comfort_addionalprompt_tests/test01_camera_side/run_gemma.sh
```

The model-specific runners are `run_experiment_qwen35.sh` and
`run_experiment_gemma.sh`. Both use the same shared setup, prompt inputs,
randomization, output directory, and resume logic.

To run inside Docker, select the model family explicitly:

```bash
QWEN_GPU=0 bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh qwen35 test01_camera_side
GEMMA_GPU=1 bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh gemma test01_camera_side
```

The Qwen runner evaluates non-thinking and then thinking. The Gemma runner does
the same independently. Existing compatible CSVs are resumed, and a completed
model is skipped before loading. `PYTHONHASHSEED=0` keeps the existing
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
test06: In the camera coordinate system, +X points to the image's right, +Y points downward, and +Z points forward into the scene. In the camera coordinate system, the person's forward direction is {...}.
```

Unlike tests 01–05, test06 is not cumulative. It deliberately excludes the
camera-side description and the object/person coordinates, isolating whether
the camera-coordinate convention plus the person's orientation is sufficient.

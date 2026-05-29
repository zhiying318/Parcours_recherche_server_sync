# WhatsUp MCQ Evaluation

This folder evaluates the validated WhatsUp human-centered images with the same
single-image long MCQ format used in COMFORT.

```bash
cd /home/zzou/whatsup_eval
python generate_image_paths.py
./run.sh
```

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

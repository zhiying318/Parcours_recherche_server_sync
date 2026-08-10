# COCO person orientation smoke-test samples

This directory contains eight visually reviewed images from the COCO 2017
validation split. Each selected person is provided as a padded bounding-box crop
because Orient Anything V2 expects one principal object per input image.

- `originals/`: unmodified COCO val2017 images.
- `person_crops/`: selected person crops used for inference.
- `person_crops_contact.jpg`: visual overview of all eight model inputs.
- `samples.json`: COCO IDs, boxes, source URLs, and approximate view hints.
- `annotations/instances_val2017.json`: official COCO instance annotations.

The `manual_view_hint` values are coarse visual notes created only to keep this
small set diverse. They are not part of COCO and must not be treated as ground
truth for quantitative evaluation.

Each manifest entry also retains the image license metadata distributed in the
official COCO annotation file.

Regenerate the selected files from the official annotations and source images:

```bash
conda activate orianyv2
python prepare_coco_person_samples.py
```

# COCO directional-object smoke-test samples

This directory contains eight visually reviewed COCO val2017 objects: bicycle,
car, motorcycle, airplane, bus, train, truck, and boat. Padded instance bounding
boxes are used because Orient Anything V2 expects one principal object per image.

- `originals/`: unmodified COCO images.
- `object_crops/`: model inputs.
- `object_crops_contact.jpg`: visual overview of the eight inputs.
- `samples.json`: COCO IDs, boxes, licenses, and coarse manual view hints.

The view hints are diversity notes, not official COCO annotations or quantitative
ground truth.

```bash
conda activate orianyv2
python prepare_coco_object_samples.py
```

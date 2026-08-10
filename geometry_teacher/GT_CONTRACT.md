# COMFORT geometry ground-truth contract (v1)

Each rendered sample is stored in its own directory:

```text
sample/
  rgb.png
  person_mask.png
  object_mask.png
  depth.npy
  camera.json
  scene_gt.json
```

All JSON matrices are row-major arrays. Floating-point geometry is stored as
`float64` in JSON. `depth.npy` is a `float32` array with the RGB image height
and width and contains positive OpenCV camera-axis depth (`z_c`), not Euclidean
ray distance. Invalid/background depth is `NaN`. Masks are single-channel PNGs
at RGB resolution: foreground is 255 and background is 0.

## Coordinate systems

The VGGT/OpenCV camera frame is right-handed:

- `+x_c`: image right
- `+y_c`: image down
- `+z_c`: camera forward

Blender cameras use local `+x` right, `+y` up, and look along local `-z`.
Given Blender's camera-to-world matrix `T_wc_blender`, the exported transform is

```text
T_cw_opencv = diag(1, -1, -1, 1) @ inverse(T_wc_blender)
```

The human coordinate frame is also right-handed:

- `+x_h`: human right
- `+y_h`: human up
- `+z_h`: human back
- human forward is therefore `-z_h`

`scene_gt.json` records semantic `right_axis`, `up_axis`, and `forward_axis`.
The rotation in `camera_to_human_transform` has rows
`[right_axis_camera, up_axis_camera, back_axis_camera]`, where
`back_axis_camera = -forward_axis_camera`. It is a proper rotation with
determinant `+1`:

```text
p_h = R_hc @ p_c + t_hc
t_hc = -R_hc @ human_origin_camera
```

For the current COMFORT configuration, the semantic world axes are:

```text
right_world   = [-1,  0,  0]
up_world      = [ 0,  0,  1]
forward_world = [ 0, -1,  0]
back_world    = [ 0,  1,  0]
```

## Centers and relation

The teacher center is the component-wise median of valid visible 3D points
selected by the corresponding binary mask. Blender object origins are retained
as separate audit fields and are never substituted for visible-point centers.

For relative position `[x_h, y_h, z_h]`, classification uses only the human
horizontal plane:

```text
if abs(x_h) > abs(z_h):
    x_h > 0 -> right
    x_h < 0 -> left
else:
    z_h < 0 -> front
    z_h >= 0 -> back
```

The exact diagonal tie belongs to the front/back branch. Every sample records
`classification_margin = abs(abs(x_h) - abs(z_h))` for later ambiguity checks.

## Required `camera.json` fields

```json
{
  "contract_version": "1.0",
  "image_size": [512, 512],
  "intrinsic_opencv": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "world_to_camera_opencv": [],
  "camera_to_world_opencv": []
}
```

## Required `scene_gt.json` fields

```json
{
  "contract_version": "1.0",
  "object_name": "chair",
  "relation": "left",
  "human_origin_world": [],
  "object_origin_world": [],
  "human_visible_center_camera": [],
  "object_visible_center_camera": [],
  "human_frame_camera": {
    "right_axis": [],
    "up_axis": [],
    "forward_axis": [],
    "back_axis": []
  },
  "camera_to_human_transform": {
    "rotation": [],
    "translation": []
  },
  "object_relative_position_human": [],
  "classification_margin": 0.0
}
```

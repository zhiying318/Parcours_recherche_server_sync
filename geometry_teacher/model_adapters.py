"""Lazy adapters for the four official geometry-teacher models.

Importing this module never initializes CUDA or downloads checkpoints.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_grounding_query(query: str) -> str:
    """Return the sentence-terminated text expected by Grounding DINO."""
    normalized = query.strip().rstrip(".").strip()
    if not normalized:
        raise ValueError("Grounding DINO query must be non-empty")
    return normalized + "."


class VggtAdapter:
    def __init__(self, model_id: str = "facebook/VGGT-1B", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None

    def predict(self, image_path: str | Path) -> dict:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import depth_to_cam_coords_points
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        if self._model is None:
            self._model = VGGT.from_pretrained(self.model_id).to(self.device).eval()
        images = load_and_preprocess_images([str(image_path)], mode="pad").to(self.device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            predictions = self._model(images)
        height, width = images.shape[-2:]
        _, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], image_size_hw=(height, width)
        )
        depth = predictions["depth"][0, 0, ..., 0].float().cpu().numpy()
        intrinsic = intrinsics[0, 0].float().cpu().numpy()
        point_map = depth_to_cam_coords_points(depth, intrinsic)
        confidence = predictions["depth_conf"][0, 0].float().cpu().numpy()
        original_size = Image.open(image_path).size
        return {
            "depth": depth,
            "depth_confidence": confidence,
            "point_map_camera": point_map,
            "intrinsic_processed": intrinsic,
            "processed_size_hw": [height, width],
            "original_size_hw": [original_size[1], original_size[0]],
        }

    def close(self) -> None:
        self._model = None
        _release_cuda()


class GroundingDinoAdapter:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._processor = None
        self._model = None

    def detect_one(self, image_path: str | Path, query: str) -> dict:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        query = normalize_grounding_query(query)
        if self._model is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            self._model = self._model.to(self.device).eval()
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results.get("text_labels", results.get("labels"))
        if len(boxes) != 1:
            raise ValueError(f"Expected exactly one detection for {query!r}, found {len(boxes)}")
        label = labels[0] if labels is not None else query
        return {"box_xyxy": boxes[0], "score": float(scores[0]), "label": str(label)}

    def close(self) -> None:
        self._processor = None
        self._model = None
        _release_cuda()


class Sam2Adapter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._predictor = None

    def segment_box(self, image_path: str | Path, box_xyxy: np.ndarray) -> np.ndarray:
        import torch
        from transformers import Sam2Model, Sam2Processor

        if self._predictor is None:
            self._processor = Sam2Processor.from_pretrained(self.model_id)
            self._predictor = (
                Sam2Model.from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )
        image = Image.open(image_path).convert("RGB")
        box = np.asarray(box_xyxy, dtype=np.float32)
        if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid SAM2 box in xyxy format: {box}")
        inputs = self._processor(
            images=image,
            input_boxes=[[box.tolist()]],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self._predictor(**inputs, multimask_output=False)
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )
        if len(masks) != 1 or tuple(masks[0].shape[:2]) != (1, 1):
            raise ValueError(
                f"Expected one image, object, and SAM2 mask, got {[tuple(x.shape) for x in masks]}"
            )
        mask = masks[0][0, 0].numpy().astype(bool)
        if mask.shape != (image.height, image.width):
            raise ValueError(
                f"SAM2 mask shape {mask.shape} differs from image {(image.height, image.width)}"
            )
        if not np.any(mask):
            raise ValueError("SAM2 returned an empty mask")
        return mask

    def close(self) -> None:
        self._processor = None
        self._predictor = None
        _release_cuda()


class ViTPoseAdapter:
    def __init__(
        self,
        model_id: str = "usyd-community/vitpose-base",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def predict_one(self, image_path: str | Path, person_box_xyxy: np.ndarray, score: float) -> np.ndarray:
        import torch
        from transformers import AutoImageProcessor, VitPoseForPoseEstimation

        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = (
                VitPoseForPoseEstimation.from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )
        image = Image.open(image_path).convert("RGB")
        box = np.asarray(person_box_xyxy, dtype=np.float32).copy()
        if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid person box in xyxy format: {box}")
        box[2:] -= box[:2]
        boxes = [box[None, :]]
        inputs = self._processor(image, boxes=boxes, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        pose_results = self._processor.post_process_pose_estimation(
            outputs,
            boxes=boxes,
        )
        if len(pose_results) != 1 or len(pose_results[0]) != 1:
            raise ValueError(f"Expected one image and one ViTPose result, got {pose_results}")
        result = pose_results[0][0]
        labels = result["labels"].detach().cpu().numpy().astype(int)
        xy = result["keypoints"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        if sorted(labels.tolist()) != list(range(17)):
            raise ValueError(f"Expected COCO keypoint labels 0..16, got {labels.tolist()}")
        keypoints = np.empty((17, 3), dtype=np.float32)
        keypoints[labels, :2] = xy
        keypoints[labels, 2] = scores
        if keypoints.shape != (17, 3):
            raise ValueError(f"Expected COCO keypoints with shape (17, 3), got {keypoints.shape}")
        return keypoints

    def close(self) -> None:
        self._processor = None
        self._model = None
        _release_cuda()


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

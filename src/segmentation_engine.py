from __future__ import annotations

from typing import List, Sequence, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Facebook SAM2 – install from https://github.com/facebookresearch/sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------------------------------------------------------------
#                           DEVICE SELECTION
# ---------------------------------------------------------------------------

def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
#                          SEGMENTATION ENGINE
# ---------------------------------------------------------------------------

BBox = Tuple[int, int, int, int]  # x, y, w, h
Mask = np.ndarray  # H×W, uint8 {0,1}


class SegmentationEngine:
    _DEFAULT_MODEL = "facebook/sam2.1-hiera-large"

    # ---------- construction ----------
    def __init__(self, device: Optional[str] = None):
        self.device: str = device or select_device()
        self._mask_generator: Optional[SAM2AutomaticMaskGenerator] = None
        self._predictor: Optional[SAM2ImagePredictor] = None

        self.original_image: Optional[Image.Image] = None
        self.segmented_masks: List[dict] = []
        self.object_info: List[Tuple[Mask, BBox]] = []

    # ---------- helpers ----------
    @staticmethod
    def _pil_to_np(img: Image.Image) -> np.ndarray:
        return np.array(img.convert("RGB"), copy=True)

    # ---------- public API ----------
    def load_image(self, img: Image.Image) -> None:
        self.original_image = img.convert("RGB")

    def segment_auto(
        self,
        *,
        mask_generator_params: Optional[dict] = None,
    ) -> List[dict]:

        if self.original_image is None:
            raise RuntimeError("load_image() must be called first")

        # lazily (re‑)build the generator so the GUI can tweak parameters
        self._build_mask_generator(mask_generator_params or {})

        img_np = self._pil_to_np(self.original_image)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.segmented_masks = self._mask_generator.generate(img_np)
            else:
                self.segmented_masks = self._mask_generator.generate(img_np)

        self.object_info = [
            (m["segmentation"].astype(np.uint8), m["bbox"]) for m in self.segmented_masks
        ]
        return self.segmented_masks

    def segment_prompt(
        self,
        *,
        points: Optional[Sequence[Tuple[int, int, int]]] = None,  # (x,y,label)
        rectangles: Optional[Sequence[Tuple[int, int, int, int]]] = None,  # x0,y0,x1,y1
    ) -> Mask:
        
        if self.original_image is None:
            raise RuntimeError("load_image() must be called first")

        # lazy predictor construction (cheaper than mask‑generator)
        if self._predictor is None:
            self._predictor = SAM2ImagePredictor.from_pretrained(
                self._DEFAULT_MODEL, device=self.device
            )

        img_np = self._pil_to_np(self.original_image)
        self._predictor.set_image(img_np)

        point_coords, point_labels = None, None
        if points:
            pts = np.array([[x, y] for x, y, _ in points], dtype=np.float32)
            lbl = np.array([lab for *_xy, lab in points], dtype=np.int64)
            point_coords, point_labels = pts, lbl

        boxes = None
        if rectangles:
            boxes = np.array(rectangles, dtype=np.float32)

        with torch.inference_mode():
            masks, _, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=boxes,
                multimask_output=False,
            )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 returned no mask for the given prompts")

        mask = np.any(masks, axis=0).astype(np.uint8) if masks.ndim == 3 else masks.astype(np.uint8)
        self.segmented_masks = [
            {
                "segmentation": mask,
                "bbox": self._mask_to_bbox(mask),
                "area": int(mask.sum()),
                "predicted_iou": 1.0,
            }
        ]
        self.object_info = [(mask, self.segmented_masks[0]["bbox"])]
        return mask

    @staticmethod
    def create_mask_overlay(
        image_hw: Tuple[int, int],
        anns: Sequence[dict],
        *,
        borders: bool = True,
    ) -> Image.Image:

        H, W = image_hw
        sorted_anns = sorted(anns, key=lambda x: x.get("area", 0), reverse=True)
        overlay = np.zeros((H, W, 4), dtype=np.float32)

        for ann in sorted_anns:
            mask = ann["segmentation"].astype(np.uint8)
            color_mask = np.random.rand(3)
            alpha_val = 0.5

            overlay[mask == 1, :3] = color_mask
            overlay[mask == 1, 3] = alpha_val

            if borders:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(
                    overlay,
                    contours,
                    -1,
                    (0.0, 0.0, 1.0, alpha_val),
                    thickness=1,
                )

        return Image.fromarray((overlay * 255).astype(np.uint8), mode="RGBA")

    @staticmethod
    def extract_object(original: Image.Image, mask: Mask) -> Image.Image:
        image_rgba = original.convert("RGBA")
        alpha_np = np.zeros(mask.shape, dtype=np.uint8)
        alpha_np[mask == 1] = 255
        alpha_mask = Image.fromarray(alpha_np, mode="L")

        transparent_bg = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
        object_rgba = Image.composite(image_rgba, transparent_bg, alpha_mask)

        ys, xs = np.where(mask == 1)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        return object_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))

    # -----------------------------------------------------------------------
    #                           internal helpers
    # -----------------------------------------------------------------------
    def _build_mask_generator(self, params: dict) -> None:
        default = dict(
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.95,
            stability_score_offset=1.0,
            crop_n_layers=1,
            box_nms_thresh=0.3,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True,
        )
        cfg = {**default, **params}
        self._mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
            self._DEFAULT_MODEL,
            device=self.device,
            **cfg,
        )

    @staticmethod
    def _mask_to_bbox(mask: Mask) -> BBox:
        ys, xs = np.where(mask == 1)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        return x_min, y_min, x_max - x_min, y_max - y_min
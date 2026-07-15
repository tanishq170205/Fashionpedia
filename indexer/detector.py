"""
Garment and person detector.

Uses Grounding DINO via the transformers library as the primary backend.
Falls back to OWL-ViT if GDINO weights cannot be loaded (e.g., on machines
where the GDINO checkpoint is unavailable or the CUDA build fails).

Two detection passes are made per image:
  1. Garment pass: detects fashion items using a broad open-vocab prompt.
  2. Person pass: detects person bounding boxes, used only for garment-to-person
     association at indexing time. Person boxes are not stored as regions.

Person-garment association is critical for compositional query correctness.
Without it, an image of two people — one in a red shirt, one in blue pants —
would incorrectly score as a full match for "red shirt AND blue pants". By
grouping garments by the person they overlap with, the reranker can enforce
that all query attributes land on the same individual.

The person detection pass uses the same detector, so the only overhead is one
extra forward pass per image. This is acceptable during offline indexing.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from PIL import Image

# Garment vocabulary for the detection prompt.
# GDINO is open-vocabulary, so this prompt seeds the search space but the
# detector can return boxes outside it. We keep the list reasonably broad to
# cover the Fashionpedia taxonomy without being exhaustive.
# Combined prompt runs garment + person detection in one forward pass,
# which roughly halves inference time on CPU compared to two separate passes.
_COMBINED_PROMPT = (
    "person . shirt . jacket . coat . pants . dress . skirt . hat . bag . shoes . tie . "
    "jeans . sweater . blazer . shorts . scarf . gloves . belt . suit . hoodie . "
    "cardigan . trousers . windbreaker . saree . kimono . vest . leggings . "
    "boots . sneakers . sandals . handbag . backpack . cap . beanie"
)

# Labels that belong to the "person" category — used to split combined results.
_PERSON_LABELS = {"person", "man", "woman", "child", "boy", "girl", "people"}


def load_detector(model_name: str = "groundingdino", device: str = "cuda") -> Any:
    """
    Load and return a detector object.

    The returned object has a single method: detect(image, prompt, threshold).
    This abstraction lets the rest of the codebase be indifferent to which
    backend is running.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Detector will run on CPU.")
        device = "cpu"

    if model_name == "groundingdino":
        try:
            return _load_gdino(device)
        except Exception as e:
            warnings.warn(
                f"Grounding DINO failed to load ({e}). Falling back to OWL-ViT.",
                stacklevel=2,
            )
            return _load_owlvit(device)

    if model_name == "owlvit":
        return _load_owlvit(device)

    raise ValueError(f"Unknown detector model: {model_name!r}. Use 'groundingdino' or 'owlvit'.")


def detect_garments_and_persons(
    detector: Any,
    image: Image.Image,
    threshold: float = 0.30,
) -> dict:
    """
    Run garment and person detection on a single PIL image.

    Returns:
        {
          "garments": [
            {
              "label": str,
              "bbox": [x1, y1, x2, y2],   # absolute pixel coords
              "score": float,
              "person_id": int,            # index into persons list, or -1
            },
            ...
          ],
          "persons": [
            {"bbox": [x1, y1, x2, y2], "score": float},
            ...
          ],
        }

    If no garments are detected, returns empty lists. The image is still indexed
    (with only its full-image embedding) so it participates in stage-1 retrieval.
    """
    # Single detection pass with combined prompt — split results by label afterward.
    all_boxes = detector.detect(image, _COMBINED_PROMPT, threshold)
    person_boxes = [b for b in all_boxes if b["label"] in _PERSON_LABELS]
    garment_boxes = [b for b in all_boxes if b["label"] not in _PERSON_LABELS]

    # Associate each garment to the most-overlapping person box.
    garments_with_persons = _assign_person_ids(garment_boxes, person_boxes)

    return {
        "garments": garments_with_persons,
        "persons": person_boxes,
    }


def _assign_person_ids(
    garments: list[dict],
    persons: list[dict],
) -> list[dict]:
    """
    For each garment box, find the person box with the highest overlap fraction
    and assign its index as person_id. Garments with no overlapping person get
    person_id = -1.

    Overlap fraction is computed as (intersection area) / (garment area), not
    IoU. This is intentional: a person box from a detector is often tight to the
    torso, meaning a long-sleeve shirt or full-length pants will extend beyond
    the person box. Using garment-relative overlap (how much of the garment is
    inside any person box) avoids falsely assigning person_id = -1 to limb-
    region garments.

    Threshold of 0.25: at least 25% of the garment box must be inside the
    person box to count as an association. Lower would allow cross-person leakage
    in crowded scenes.
    """
    OVERLAP_THRESHOLD = 0.25

    for garment in garments:
        best_person_id = -1
        best_overlap = 0.0
        gx1, gy1, gx2, gy2 = garment["bbox"]
        g_area = max(1, (gx2 - gx1) * (gy2 - gy1))

        for pid, person in enumerate(persons):
            px1, py1, px2, py2 = person["bbox"]

            # Intersection rectangle.
            ix1 = max(gx1, px1)
            iy1 = max(gy1, py1)
            ix2 = min(gx2, px2)
            iy2 = min(gy2, py2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue  # no overlap

            inter_area = (ix2 - ix1) * (iy2 - iy1)
            overlap_fraction = inter_area / g_area

            if overlap_fraction > best_overlap:
                best_overlap = overlap_fraction
                best_person_id = pid

        garment["person_id"] = best_person_id if best_overlap >= OVERLAP_THRESHOLD else -1

    return garments


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _GDINODetector:
    """
    Grounding DINO detector using the transformers pipeline.

    The transformers implementation is preferred over the standalone
    groundingdino-py package because it is pip-installable without a custom
    CUDA build step, making it reliably cross-platform on Windows.
    """

    def __init__(self, device: str) -> None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-tiny"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model.to(device).eval()
        self._device = device
        print(f"Loaded Grounding DINO ({model_id}) on {device}.")

    def detect(self, image: Image.Image, prompt: str, threshold: float) -> list[dict]:
        """
        Run detection and return boxes above the threshold.

        prompt: period-separated class names, e.g. "shirt . pants . dress"
        Returns list of {"label", "bbox", "score"} with absolute pixel coords.
        """
        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process to absolute bbox coords.
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,          # renamed from box_threshold in transformers 5.x
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )

        detections = []
        if results:
            result = results[0]
            # Use "text_labels" (string names) not "labels" (integer ids) — changed in transformers 5.x.
            label_key = "text_labels" if "text_labels" in result else "labels"
            for box, score, label in zip(
                result["boxes"], result["scores"], result[label_key]
            ):
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    "label": label.strip().lower(),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "score": round(float(score), 4),
                })

        return detections


class _OWLViTDetector:
    """
    OWL-ViT fallback detector.

    OWL-ViT takes explicit text queries per class rather than a period-separated
    prompt. We split the prompt on periods and pass each token as a separate
    query string. This is slightly less efficient than GDINO's single-pass
    open-vocab detection but is more portable.
    """

    def __init__(self, device: str) -> None:
        from transformers import pipeline as hf_pipeline

        self._pipe = hf_pipeline(
            "zero-shot-object-detection",
            model="google/owlvit-base-patch32",
            device=0 if device == "cuda" else -1,
        )
        self._device = device
        print(f"Loaded OWL-ViT (google/owlvit-base-patch32) on {device}.")

    def detect(self, image: Image.Image, prompt: str, threshold: float) -> list[dict]:
        labels = [t.strip() for t in prompt.split(".") if t.strip()]
        predictions = self._pipe(image, candidate_labels=labels)

        detections = []
        for pred in predictions:
            if pred["score"] < threshold:
                continue
            box = pred["box"]
            detections.append({
                "label": pred["label"].strip().lower(),
                "bbox": [
                    round(box["xmin"]),
                    round(box["ymin"]),
                    round(box["xmax"]),
                    round(box["ymax"]),
                ],
                "score": round(float(pred["score"]), 4),
            })

        return detections


def _load_gdino(device: str) -> _GDINODetector:
    return _GDINODetector(device)


def _load_owlvit(device: str) -> _OWLViTDetector:
    return _OWLViTDetector(device)

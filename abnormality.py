import json
import os

import cv2
import numpy as np

from segmentation import segment_image

CALIBRATION_FILE = "calibration.json"
DEFAULT_LEVEL = 150.0     # only used if calibrate has not been run
CALIBRATION_PERCENTILE = 95
MIN_BLOB_AREA = 80        # px at 224x224; removes rib fragments / specks
RIB_SUPPRESS_KERNEL = 9   # opening size; ribs are thin, opacities are wide


def load_abnormal_level():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return float(json.load(f)["abnormal_level"])
    return DEFAULT_LEVEL


def calibrate_abnormal_level(dataset_path, split="train", max_images=600):
    """Fix ONE absolute 'too bright for lung' intensity from NORMAL training
    images (95th percentile of normal inner-lung pixels).

    The old code used the 85th percentile of every image's own lung pixels,
    which flags ~15% of ANY lung, healthy or not. An absolute level learned
    from normals lets healthy lungs score low and opacified lungs score high.
    """
    from preprocessing import preprocess_image

    folder = os.path.join(dataset_path, split, "NORMAL")
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )[:max_images]

    samples = []
    for name in files:
        img = preprocess_image(os.path.join(folder, name))
        if img is None:
            continue
        seg = segment_image(img)
        if seg is None:
            continue
        smooth = cv2.GaussianBlur(seg["image"], (9, 9), 0)
        px = smooth[seg["mask"] > 0]
        samples.append(px[:: max(1, len(px) // 2000)])

    level = float(np.percentile(np.concatenate(samples), CALIBRATION_PERCENTILE))

    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"abnormal_level": level, "images_used": len(samples)}, f)

    return level


def detect_abnormal_regions(image, lung_mask, level=None):
    if level is None:
        level = load_abnormal_level()

    if cv2.countNonZero(lung_mask) == 0:
        return None

    smooth = cv2.GaussianBlur(image, (9, 9), 0)

    mask = ((smooth >= level) & (lung_mask > 0)).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (RIB_SUPPRESS_KERNEL, RIB_SUPPRESS_KERNEL)
        )
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
            cleaned[labels == i] = 255

    cleaned[lung_mask == 0] = 0
    return cleaned


def calculate_abnormality(image, lung_mask, level=None):
    abnormal_mask = detect_abnormal_regions(image, lung_mask, level)

    if abnormal_mask is None:
        return None

    lung_pixels = cv2.countNonZero(lung_mask)
    abnormal_pixels = cv2.countNonZero(abnormal_mask)

    n, _, stats, _ = cv2.connectedComponentsWithStats(abnormal_mask)
    largest = max(
        (stats[i, cv2.CC_STAT_AREA] for i in range(1, n)), default=0
    )

    return {
        "lung_area": int(lung_pixels),
        "abnormal_area": int(abnormal_pixels),
        "affected_area_percentage": float(abnormal_pixels / lung_pixels * 100.0),
        "largest_blob_percentage": float(largest / lung_pixels * 100.0),
        "blob_count": int(max(n - 1, 0)),
        "abnormal_mask": abnormal_mask
    }


def analyze_image(image):
    """image: preprocessed uint8 grayscale."""
    seg = segment_image(image)
    if seg is None:
        return None

    ab = calculate_abnormality(seg["image"], seg["mask"])
    if ab is None:
        return None

    return {
        "image": seg["image"],
        "lung_mask": seg["mask"],
        "lung_outline": seg["outline"],
        "lung_roi": seg["roi"],
        **ab
    }
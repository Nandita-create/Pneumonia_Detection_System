import os

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops

from preprocessing import preprocess_image, DATASET_PATH, LABELS
from abnormality import (
    calculate_abnormality,
    calibrate_abnormal_level,
    load_abnormal_level
)
from segmentation import segment_image


FEATURE_COLUMNS = [
    "mean_intensity", "std_intensity", "variance",
    "min_intensity", "max_intensity",
    "p10_intensity", "p50_intensity", "p90_intensity",
    "skewness", "kurtosis",
    "dark_pixel_percentage", "bright_pixel_percentage",
    "texture_contrast", "texture_correlation",
    "texture_energy", "texture_homogeneity",
    "edge_density",
    "lung_area_fraction",
    "lr_intensity_diff", "upper_lower_diff",
    "affected_area_percentage", "largest_blob_percentage", "blob_count"
]


def extract_intensity_features(image, mask):
    px = image[mask > 0].astype(np.float32)
    p10, p50, p90 = np.percentile(px, [10, 50, 90])

    return {
        "mean_intensity": float(px.mean()),
        "std_intensity": float(px.std()),
        "variance": float(px.var()),
        "min_intensity": float(px.min()),
        "max_intensity": float(px.max()),
        "p10_intensity": float(p10),
        "p50_intensity": float(p50),
        "p90_intensity": float(p90),
        "skewness": float(skew(px)),
        "kurtosis": float(kurtosis(px)),
    }


def extract_histogram_features(image, mask, level):
    px = image[mask > 0]
    n = len(px)
    return {
        "dark_pixel_percentage": float(np.sum(px < 50) / n * 100),
        "bright_pixel_percentage": float(np.sum(px >= level) / n * 100),
    }


def extract_texture_features(image, mask):
    """GLCM over lung pixels only: level 0 is reserved for 'outside mask'
    and its rows/cols are zeroed so the lung boundary does not pollute it."""
    q = (image.astype(np.float32) / 256.0 * 15).astype(np.uint8) + 1
    q[mask == 0] = 0

    glcm = graycomatrix(
        q, distances=[2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16, symmetric=True, normed=False
    ).astype(np.float64)

    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0

    sums = glcm.sum(axis=(0, 1), keepdims=True)
    sums[sums == 0] = 1
    glcm /= sums

    def prop(name):
        return float(np.mean(graycoprops(glcm, name)))

    return {
        "texture_contrast": prop("contrast"),
        "texture_correlation": prop("correlation"),
        "texture_energy": prop("energy"),
        "texture_homogeneity": prop("homogeneity"),
    }


def extract_edge_features(image, mask):
    edges = cv2.Canny(image, 50, 150)
    edges[mask == 0] = 0
    return {"edge_density": float(np.sum(edges > 0) / np.sum(mask > 0))}


def extract_regional_features(image, mask):
    h, w = mask.shape
    total = float(h * w)

    left = mask.copy();  left[:, w // 2:] = 0
    right = mask.copy(); right[:, :w // 2] = 0

    ys, _ = np.where(mask > 0)
    mid_y = int(np.median(ys))
    upper = mask.copy(); upper[mid_y:, :] = 0
    lower = mask.copy(); lower[:mid_y, :] = 0

    def m(region):
        return float(image[region > 0].mean()) if np.any(region) else 0.0

    return {
        "lung_area_fraction": float(np.sum(mask > 0) / total),
        "lr_intensity_diff": abs(m(left) - m(right)),
        "upper_lower_diff": m(lower) - m(upper),
    }


def extract_all_features(image, level=None):
    """image: PREPROCESSED uint8 grayscale (preprocess_array output)."""
    if level is None:
        level = load_abnormal_level()

    seg = segment_image(image)
    if seg is None:
        return None

    img, mask = seg["image"], seg["mask"]

    ab = calculate_abnormality(img, mask, level)
    if ab is None:
        return None

    f = {}
    f.update(extract_intensity_features(img, mask))
    f.update(extract_histogram_features(img, mask, level))
    f.update(extract_texture_features(img, mask))
    f.update(extract_edge_features(img, mask))
    f.update(extract_regional_features(img, mask))
    f["affected_area_percentage"] = ab["affected_area_percentage"]
    f["largest_blob_percentage"] = ab["largest_blob_percentage"]
    f["blob_count"] = ab["blob_count"]

    return f


def extract_features_from_path(image_path, level=None):
    image = preprocess_image(image_path)   # already uint8 0-255
    if image is None:
        return None
    return extract_all_features(image, level)


def _one(path, file_name, class_name, label, split, level):
    f = extract_features_from_path(path, level)
    if f is None:
        return None
    f.update(image_name=file_name, **{"class": class_name},
             label=label, split=split)
    return f


def process_dataset(dataset_path, split, level):
    jobs = []
    for class_name, label in LABELS.items():
        folder = os.path.join(dataset_path, split, class_name)
        if not os.path.exists(folder):
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                jobs.append((os.path.join(folder, fn), fn, class_name, label))

    rows = Parallel(n_jobs=-1)(
        delayed(_one)(p, fn, c, l, split, level) for p, fn, c, l in jobs
    )

    ok = [r for r in rows if r is not None]
    print(f"[{split}] segmented {len(ok)}/{len(jobs)} images "
          f"({len(jobs) - len(ok)} failed and were skipped)")
    return pd.DataFrame(ok)


def build_feature_dataset():
    level = calibrate_abnormal_level(DATASET_PATH)
    print(f"Calibrated abnormal intensity level: {level:.1f}")

    train_df = process_dataset(DATASET_PATH, "train", level)
    test_df = process_dataset(DATASET_PATH, "test", level)
    return pd.concat([train_df, test_df], ignore_index=True)


if __name__ == "__main__":
    df = build_feature_dataset()
    df.to_csv("feature_data.csv", index=False)
    print("Feature dataset created:", df.shape)
    print(df.groupby(["split", "class"]).size())
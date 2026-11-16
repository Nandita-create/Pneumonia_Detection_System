import cv2
import numpy as np

from preprocessing import preprocess_image

# ---- tunable geometry (fractions of each lung's bounding box) ----------
MEDIAL_CUT = 0.18     # strip next to the spine / mediastinum / heart
BOTTOM_CUT = 0.15     # diaphragm + costophrenic angle
TOP_CUT = 0.05        # clavicle / apex edge
ERODE_PX = 3          # removes the rib-cage / pleural edge band
MIN_CORE_FRACTION = 0.05
MAX_CORE_FRACTION = 0.45


def _elliptical(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _candidate_lung_blobs(binary, shape):
    """Return (left, right) contours from a binary 'dark region' mask,
    or None if the geometry is not plausible for two lungs."""
    h, w = shape
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _elliptical(5), iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _elliptical(5), iterations=1)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    left, right = [], []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.02 * h * w or area > 0.40 * h * w:
            continue

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

        # Background / air outside the body lives at the image edges
        if not (0.12 * w < cx < 0.88 * w and 0.18 * h < cy < 0.92 * h):
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if y <= 1:  # touches the top edge -> not a lung
            continue

        (left if cx < w / 2 else right).append((area, c))

    if not left or not right:
        return None

    left_c = max(left, key=lambda t: t[0])
    right_c = max(right, key=lambda t: t[0])

    ratio = min(left_c[0], right_c[0]) / max(left_c[0], right_c[0])
    if ratio < 0.30:
        return None

    return left_c[1], right_c[1]


def _core_region(hull_mask, is_left):
    """Keep only the 'inner lung field': drop pleural edge, mediastinal
    strip, diaphragm band and apex from one lung."""
    ys, xs = np.where(hull_mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    core = cv2.erode(hull_mask, _elliptical(2 * ERODE_PX + 1))

    # image-left lung -> its medial side is on the RIGHT of the bbox
    if is_left:
        core[:, int(x1 - MEDIAL_CUT * bw):] = 0
    else:
        core[:, :int(x0 + MEDIAL_CUT * bw)] = 0

    core[int(y1 - BOTTOM_CUT * bh):, :] = 0
    core[:int(y0 + TOP_CUT * bh), :] = 0

    return core


def create_lung_mask(image):
    """Segment the two lung fields.

    Lungs are DARK on a chest X-ray (air). We threshold for dark regions,
    discard background, keep one blob per side, take the convex hull (so
    consolidated/bright pneumonia areas stay inside the lung), then remove
    the rib-cage edge, mediastinum and diaphragm bands.

    Returns (core_mask, outline_mask) or (None, None) if segmentation fails.
    """
    image = image.astype(np.uint8)
    h, w = image.shape
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    otsu, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    for factor in (1.0, 0.9, 0.8, 0.7, 1.1, 0.6):
        level = otsu * factor
        binary = (blurred < level).astype(np.uint8) * 255

        blobs = _candidate_lung_blobs(binary, (h, w))
        if blobs is None:
            continue

        outline = np.zeros((h, w), np.uint8)
        core = np.zeros((h, w), np.uint8)

        for i, contour in enumerate(blobs):
            hull = cv2.convexHull(contour)
            hull_mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 255)

            outline = cv2.bitwise_or(outline, hull_mask)
            core = cv2.bitwise_or(core, _core_region(hull_mask, is_left=(i == 0)))

        frac = cv2.countNonZero(core) / float(h * w)
        if MIN_CORE_FRACTION <= frac <= MAX_CORE_FRACTION:
            return core, outline

    return None, None


def extract_lung_roi(image, lung_mask):
    image_uint8 = image.astype(np.uint8)
    return cv2.bitwise_and(image_uint8, image_uint8, mask=lung_mask)


def segment_image(image):
    """image: preprocessed uint8 grayscale. Returns None if lungs not found."""
    image = image.astype(np.uint8)

    core, outline = create_lung_mask(image)

    if core is None:
        return None

    return {
        "image": image,
        "mask": core,            # inner lung field used for ALL measurements
        "outline": outline,      # full lung outline (for display)
        "roi": extract_lung_roi(image, core)
    }


def segment_lungs(image_path):
    image = preprocess_image(image_path)

    if image is None:
        return None

    return segment_image(image)
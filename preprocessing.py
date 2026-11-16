import os
import cv2
import numpy as np

# Change this (or set the CXR_DATASET env variable) to your dataset folder
DATASET_PATH = os.environ.get(
    "CXR_DATASET",
    "C:/Users/Nandita/Downloads/ImageProcessingCP/chest_xray"
)

LABELS = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

IMAGE_SIZE = (224, 224)


def preprocess_array(image):
    """Grayscale -> resize -> blur -> CLAHE. Returns uint8 (0-255).

    NOTE: the image is NOT inverted and NOT rescaled anywhere else.
    Training and the Streamlit app must both use exactly this function.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    return image.astype(np.uint8)


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    return preprocess_array(image)
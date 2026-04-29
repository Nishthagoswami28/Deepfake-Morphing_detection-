import glob
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "Human Faces Dataset"
CLASS_FOLDERS = {
    "Deepfake": DATASET_ROOT / "AI-Generated Images",
    "Morphing Attack": DATASET_ROOT / "morphed-im",
    "Real": DATASET_ROOT / "Real Images",
}
CACHE_PATH = Path(__file__).with_name("hybrid_centroids.npz")
IMAGE_SIZE = (224, 224)

_FEATURE_STATS = None
_CENTROIDS = None


def _extract_features(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edge_density = cv2.Canny(gray, 80, 160).mean() / 255.0

    fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    magnitude = np.log1p(np.abs(fft))
    height, width = magnitude.shape
    cy, cx = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    high_freq = magnitude[radius > 40].mean()
    low_freq = magnitude[radius <= 40].mean()

    features = [
        float(laplacian_var),
        float(edge_density),
        float(high_freq),
        float(low_freq),
        float(gray.mean()),
        float(gray.std()),
    ]

    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        features.extend(hist.tolist())

    return np.array(features, dtype=np.float32)


def _iter_images(folder):
    patterns = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(folder / pattern)))
    return sorted(files)


def _build_centroids():
    samples = {}

    for label, folder in CLASS_FOLDERS.items():
        vectors = []
        for image_path in _iter_images(folder)[:250]:
            image = cv2.imread(image_path)
            if image is None:
                continue
            resized = cv2.resize(image, IMAGE_SIZE)
            vectors.append(_extract_features(resized))
        if not vectors:
            continue
        samples[label] = np.stack(vectors)

    if len(samples) < 2:
        raise RuntimeError("Not enough dataset folders with images to build the hybrid classifier.")

    all_vectors = np.concatenate(list(samples.values()), axis=0)
    mean = all_vectors.mean(axis=0)
    std = all_vectors.std(axis=0) + 1e-6
    centroids = {}

    for label, vectors in samples.items():
        centroids[label] = ((vectors - mean) / std).mean(axis=0)

    np.savez(
        CACHE_PATH,
        mean=mean,
        std=std,
        labels=np.array(list(centroids.keys())),
        centroids=np.stack([centroids[label] for label in centroids]),
    )

    return mean, std, centroids


def _load_classifier():
    global _FEATURE_STATS, _CENTROIDS

    if _FEATURE_STATS is not None and _CENTROIDS is not None:
        return _FEATURE_STATS, _CENTROIDS

    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH, allow_pickle=True)
        mean = cache["mean"]
        std = cache["std"]
        labels = cache["labels"].tolist()
        centroid_values = cache["centroids"]
        centroids = {label: centroid_values[idx] for idx, label in enumerate(labels)}
    else:
        mean, std, centroids = _build_centroids()

    _FEATURE_STATS = (mean, std)
    _CENTROIDS = centroids
    return _FEATURE_STATS, _CENTROIDS


def predict_image(face):
    if face is None:
        return "No Face Detected", 0

    (mean, std), centroids = _load_classifier()
    resized = cv2.resize(face, IMAGE_SIZE)
    vector = _extract_features(resized)
    normalized = (vector - mean) / std

    distances = {
        label: float(np.linalg.norm(normalized - centroid))
        for label, centroid in centroids.items()
    }
    ordered = sorted(distances.items(), key=lambda item: item[1])
    best_label, best_distance = ordered[0]
    second_distance = ordered[1][1] if len(ordered) > 1 else best_distance + 1.0

    scores = {
        label: np.exp(-distance)
        for label, distance in distances.items()
    }
    total = sum(scores.values()) + 1e-8
    confidence = scores[best_label] / total

    if second_distance > 0:
        confidence *= min(1.0, second_distance / max(best_distance, 1e-6))

    return best_label, round(float(confidence) * 100, 2)

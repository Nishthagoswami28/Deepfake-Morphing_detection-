import glob
from pathlib import Path

import cv2
import numpy as np

from model.morph_detection import detect_morph

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "Human Faces Dataset"
CLASS_FOLDERS = {
    "Deepfake": DATASET_ROOT / "AI-Generated Images",
    "Morphing Attack": DATASET_ROOT / "morphed-im",
    "Real": DATASET_ROOT / "Real Images",
}
CACHE_PATH = Path(__file__).with_name("hybrid_centroids.npz")
REFERENCE_CACHE_PATH = Path(__file__).with_name("hybrid_reference_vectors.npz")
IMAGE_SIZE = (224, 224)

_FEATURE_STATS = None
_CENTROIDS = None
_REFERENCE_VECTORS = None
_REFERENCE_LABELS = None


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


def _prepare_training_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        image = image[y:y + h, x:x + w]

    return cv2.resize(image, IMAGE_SIZE)


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


def _build_reference_vectors(mean, std):
    vectors = []
    labels = []

    for label, folder in CLASS_FOLDERS.items():
        for image_path in _iter_images(folder)[:250]:
            image = cv2.imread(image_path)
            if image is None:
                continue
            prepared = _prepare_training_image(image)
            vector = (_extract_features(prepared) - mean) / std
            vectors.append(vector)
            labels.append(label)

    if not vectors:
        raise RuntimeError("Not enough dataset images to build the reference classifier.")

    vectors = np.stack(vectors)
    labels = np.array(labels)
    np.savez(REFERENCE_CACHE_PATH, vectors=vectors, labels=labels)
    return vectors, labels


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


def _load_reference_classifier(mean, std):
    global _REFERENCE_VECTORS, _REFERENCE_LABELS

    if _REFERENCE_VECTORS is not None and _REFERENCE_LABELS is not None:
        return _REFERENCE_VECTORS, _REFERENCE_LABELS

    if REFERENCE_CACHE_PATH.exists():
        cache = np.load(REFERENCE_CACHE_PATH, allow_pickle=True)
        vectors = cache["vectors"]
        labels = cache["labels"]
    else:
        vectors, labels = _build_reference_vectors(mean, std)

    _REFERENCE_VECTORS = vectors
    _REFERENCE_LABELS = labels
    return _REFERENCE_VECTORS, _REFERENCE_LABELS


def _nearest_reference_label(normalized):
    (mean, std), _ = _load_classifier()
    vectors, labels = _load_reference_classifier(mean, std)
    distances = np.linalg.norm(vectors - normalized, axis=1)
    nearest_indices = np.argsort(distances)[:5]
    nearest_labels = labels[nearest_indices]
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)
    vote_label = unique_labels[np.argmax(counts)]
    nearest_distance = float(distances[nearest_indices[0]])
    nearest_label = str(labels[nearest_indices[0]])
    return nearest_label, str(vote_label), nearest_distance


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

    confidence_percent = round(float(confidence) * 100, 2)
    nearest_label, vote_label, nearest_distance = _nearest_reference_label(normalized)

    morph_score = detect_morph(resized)

    if nearest_label != best_label and nearest_distance < 1.0 and confidence_percent < 65:
        hybrid_confidence = min(90.0, confidence_percent + 20.0)
        return nearest_label, round(hybrid_confidence, 2)

    if vote_label != best_label and confidence_percent < 55:
        return vote_label, confidence_percent

    if best_label == "Deepfake" and morph_score >= 0.75 and confidence_percent < 70:
        return "Morphing Attack", confidence_percent

    return best_label, confidence_percent

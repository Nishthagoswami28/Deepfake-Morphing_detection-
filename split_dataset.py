import random
import shutil
from pathlib import Path

random.seed(42)

project_root = Path(__file__).resolve().parent

source_dirs = {
    "real": project_root / "dataset" / "Human Faces Dataset" / "Real Images",
    "ai_generated": project_root / "dataset" / "Human Faces Dataset" / "AI-Generated Images",
    "morphed": project_root / "dataset" / "Human Faces Dataset" / "morphed-im",
}

target_base = project_root / "dataset" / "hybrid_faces"

split_ratio = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

for class_name, source_dir in source_dirs.items():
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(source_dir.glob(ext))

    images = list(images)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratio["train"])
    val_end = train_end + int(total * split_ratio["val"])

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, split_images in splits.items():
        target_dir = target_base / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for index, img_path in enumerate(split_images):
            target_dir.mkdir(parents=True, exist_ok=True)
            new_name = f"{class_name}_{index:06d}{img_path.suffix.lower()}"
            shutil.copy2(img_path, target_dir / new_name)

print("Dataset split completed.")

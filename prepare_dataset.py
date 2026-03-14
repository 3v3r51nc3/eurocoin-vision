import random
import shutil
from pathlib import Path

random.seed(52)

BASE_DIR = Path(__file__).resolve().parent

src_images = BASE_DIR / "data_raw/images"
src_labels = BASE_DIR / "data_raw/labels"

out_root = BASE_DIR / "dataset"
out_images_train = out_root / "images/train"
out_images_val = out_root / "images/val"
out_labels_train = out_root / "labels/train"
out_labels_val = out_root / "labels/val"

for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
    p.mkdir(parents=True, exist_ok=True)

class_names = [
    "1_cent",
    "2_cent",
    "5_cent",
    "10_cent",
    "20_cent",
    "50_cent",
    "1_euro",
    "2_euro",
]

if not src_images.exists():
    raise FileNotFoundError(f"Images folder not found: {src_images}")

if not src_labels.exists():
    raise FileNotFoundError(f"Labels folder not found: {src_labels}")

image_files = sorted([
    p for p in src_images.iterdir()
    if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

if not image_files:
    raise ValueError(f"No images found in: {src_images}")

random.shuffle(image_files)

split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]


def copy_set(files, img_dst, lbl_dst):
    copied = 0
    skipped = 0

    for img_path in files:
        label_path = src_labels / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"Label missing for {img_path.name}")
            skipped += 1
            continue

        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(label_path, lbl_dst / label_path.name)
        copied += 1

    return copied, skipped


train_copied, train_skipped = copy_set(train_files, out_images_train, out_labels_train)
val_copied, val_skipped = copy_set(val_files, out_images_val, out_labels_val)

yaml_path = out_root / "data.yaml"

yaml_lines = [
    "train: images/train",
    "val: images/val",
    "",
    "names:",
]

for i, name in enumerate(class_names):
    yaml_lines.append(f"  {i}: {name}")

yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")

print("Done.")
print(f"Base dir: {BASE_DIR}")
print(f"Images dir: {src_images}")
print(f"Labels dir: {src_labels}")
print(f"Train copied: {train_copied}, skipped: {train_skipped}")
print(f"Val copied: {val_copied}, skipped: {val_skipped}")
print(f"YAML saved to: {yaml_path}")
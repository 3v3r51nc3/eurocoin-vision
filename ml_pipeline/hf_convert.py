from pathlib import Path
import json
from PIL import Image

ROOT = Path(r"\data_raw")
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
CLASSES = ROOT / "classes.txt"

HF_DIR = ROOT / "hf"
LS_DIR = ROOT / "label_studio"
HF_DIR.mkdir(exist_ok=True)
LS_DIR.mkdir(exist_ok=True)

class_names = [line.strip() for line in CLASSES.read_text(encoding="utf-8").splitlines() if line.strip()]

image_exts = {".jpg", ".jpeg", ".png", ".webp"}

hf_lines = []
ls_lines = []

def yolo_to_xywh_pixels(cx, cy, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x = (cx * img_w) - bw / 2
    y = (cy * img_h) - bh / 2
    return x, y, bw, bh

for img_path in sorted([p for p in IMAGES.iterdir() if p.suffix.lower() in image_exts]):
    label_path = LABELS / f"{img_path.stem}.txt"
    with Image.open(img_path) as img:
        img_w, img_h = img.size

    hf_bboxes = []
    hf_categories = []
    ls_results = []

    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id_s, cx_s, cy_s, w_s, h_s = line.split()
            cls_id = int(cls_id_s)
            cx, cy, w, h = map(float, (cx_s, cy_s, w_s, h_s))

            x, y, bw, bh = yolo_to_xywh_pixels(cx, cy, w, h, img_w, img_h)

            hf_bboxes.append([x, y, bw, bh])
            hf_categories.append(cls_id)

            ls_results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": (x / img_w) * 100.0,
                    "y": (y / img_h) * 100.0,
                    "width": (bw / img_w) * 100.0,
                    "height": (bh / img_h) * 100.0,
                    "rotation": 0,
                    "rectanglelabels": [class_names[cls_id]]
                }
            })

    hf_record = {
        "file_name": f"{img_path.name}",
        "objects": {
            "bbox": hf_bboxes,
            "category": hf_categories
        }
    }
    hf_lines.append(json.dumps(hf_record, ensure_ascii=False))

    ls_record = {
        "data": {
            "image": f"file:///{img_path.as_posix()}"
        },
        "annotations": [
            {
                "result": ls_results
            }
        ]
    }
    ls_lines.append(json.dumps(ls_record, ensure_ascii=False))

(HF_DIR / "metadata.jsonl").write_text("\n".join(hf_lines), encoding="utf-8")
(LS_DIR / "tasks.jsonl").write_text("\n".join(ls_lines), encoding="utf-8")

label_config = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
%s
  </RectangleLabels>
</View>
""" % "\n".join([f'    <Label value="{name}"/>' for name in class_names])

(LS_DIR / "label_config.xml").write_text(label_config, encoding="utf-8")

print("Done:")
print(HF_DIR / "metadata.jsonl")
print(LS_DIR / "tasks.jsonl")
print(LS_DIR / "label_config.xml")
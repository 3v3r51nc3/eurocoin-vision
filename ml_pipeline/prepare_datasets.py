from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ProjectLayout:
    root_dir: Path

    @classmethod
    def from_current_file(cls) -> "ProjectLayout":
        return cls(root_dir=Path(__file__).resolve().parent)

    @property
    def raw_root(self) -> Path:
        return self.root_dir / "data_raw"

    @property
    def raw_images_dir(self) -> Path:
        return self.raw_root / "images"

    @property
    def raw_labels_dir(self) -> Path:
        return self.raw_root / "labels"

    @property
    def raw_notes_path(self) -> Path:
        return self.raw_root / "notes.json"

    @property
    def datasets_root(self) -> Path:
        return self.root_dir / "datasets"

    @property
    def classification_root(self) -> Path:
        return self.datasets_root


@dataclass(frozen=True)
class DatasetPreparationConfig:
    layout: ProjectLayout
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    shared_stage_split: bool
    image_extensions: frozenset[str]

    @classmethod
    def default(cls) -> "DatasetPreparationConfig":
        return cls(
            layout=ProjectLayout.from_current_file(),
            seed=52,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            shared_stage_split=False,
            image_extensions=frozenset({".jpg", ".jpeg", ".png", ".webp"}),
        )


@dataclass(frozen=True)
class StageConfig:
    name: str
    class_names: list[str]
    name_to_stage_id: dict[str, int]

    def stage_id_for(self, class_name: str) -> int:
        return self.name_to_stage_id[class_name]


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class ParsedLabel:
    source_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def yolo_coords(self) -> list[str]:
        return [
            f"{self.x_center:.16f}".rstrip("0").rstrip("."),
            f"{self.y_center:.16f}".rstrip("0").rstrip("."),
            f"{self.width:.16f}".rstrip("0").rstrip("."),
            f"{self.height:.16f}".rstrip("0").rstrip("."),
        ]


@dataclass(frozen=True)
class SampleInventory:
    matched_samples: list[DatasetSample]
    missing_labels: list[Path]
    missing_images: list[Path]


@dataclass(frozen=True)
class DatasetSplit:
    train_samples: list[DatasetSample]
    val_samples: list[DatasetSample]
    test_samples: list[DatasetSample]


@dataclass(frozen=True)
class SourceClassCatalog:
    id_to_name: dict[int, str]
    class_names: list[str]

    @classmethod
    def load(cls, notes_path: Path) -> "SourceClassCatalog":
        if not notes_path.exists():
            raise FileNotFoundError(f"Source metadata file not found: {notes_path}")

        notes = json.loads(notes_path.read_text(encoding="utf-8"))
        categories = notes.get("categories", [])
        if not categories:
            raise ValueError(f"No categories found in {notes_path}")

        id_to_name: dict[int, str] = {}
        class_names: list[str] = []
        for category in sorted(categories, key=lambda item: item["id"]):
            category_id = int(category["id"])
            category_name = str(category["name"])
            id_to_name[category_id] = category_name
            class_names.append(category_name)

        return cls(id_to_name=id_to_name, class_names=class_names)

    def name_for_id(self, class_id: int) -> str:
        if class_id not in self.id_to_name:
            raise ValueError(f"Unknown class id: {class_id}")
        return self.id_to_name[class_id]


@dataclass(frozen=True)
class StageExportSummary:
    output_dir: Path
    train_counts: Counter
    val_counts: Counter
    test_counts: Counter


@dataclass(frozen=True)
class ClassificationExportSummary:
    output_dir: Path
    train_counts: Counter
    val_counts: Counter
    test_counts: Counter


@dataclass(frozen=True)
class DatasetPreparationReport:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    shared_stage_split: bool
    stage_splits: dict[str, DatasetSplit]
    inventory: SampleInventory
    stage_summaries: dict[str, StageExportSummary]
    classification_summaries: dict[str, ClassificationExportSummary]

    def print_console(self) -> None:
        print(f"Seed: {self.seed}")
        print(f"Train ratio: {self.train_ratio}")
        print(f"Val ratio: {self.val_ratio}")
        print(f"Test ratio: {self.test_ratio}")
        print(f"Shared split across stages: {self.shared_stage_split}")
        stage1_split = self.stage_splits["stage1"]
        print(
            "Matched samples: "
            f"{len(stage1_split.train_samples) + len(stage1_split.val_samples) + len(stage1_split.test_samples)}"
        )
        print(f"Stage1 train samples: {len(stage1_split.train_samples)}")
        print(f"Stage1 val samples: {len(stage1_split.val_samples)}")
        print(f"Stage1 test samples: {len(stage1_split.test_samples)}")
        print(f"Missing labels: {len(self.inventory.missing_labels)}")
        print(f"Missing images: {len(self.inventory.missing_images)}")

        if self.inventory.missing_labels:
            print("Images without labels:")
            for path in self.inventory.missing_labels:
                print(f"  - {path.name}")

        if self.inventory.missing_images:
            print("Labels without images:")
            for path in self.inventory.missing_images:
                print(f"  - {path.name}")

        for stage_name, summary in self.stage_summaries.items():
            print("")
            print(f"{stage_name}:")
            print(f"  output: {summary.output_dir}")
            print(f"  train boxes: {sum(summary.train_counts.values())}")
            print(f"  val boxes: {sum(summary.val_counts.values())}")
            print(f"  test boxes: {sum(summary.test_counts.values())}")
            print("  train classes:")
            for class_name, count in summary.train_counts.items():
                print(f"    - {class_name}: {count}")
            print("  val classes:")
            for class_name, count in summary.val_counts.items():
                print(f"    - {class_name}: {count}")
            print("  test classes:")
            for class_name, count in summary.test_counts.items():
                print(f"    - {class_name}: {count}")

        for dataset_name, summary in self.classification_summaries.items():
            print("")
            print(f"{dataset_name}:")
            print(f"  output: {summary.output_dir}")
            print(f"  train crops: {sum(summary.train_counts.values())}")
            print(f"  val crops: {sum(summary.val_counts.values())}")
            print(f"  test crops: {sum(summary.test_counts.values())}")
            print("  train classes:")
            for class_name, count in summary.train_counts.items():
                print(f"    - {class_name}: {count}")
            print("  val classes:")
            for class_name, count in summary.val_counts.items():
                print(f"    - {class_name}: {count}")
            print("  test classes:")
            for class_name, count in summary.test_counts.items():
                print(f"    - {class_name}: {count}")


class SourceDatasetScanner:
    def __init__(self, config: DatasetPreparationConfig) -> None:
        self._config = config

    def gather_samples(self) -> SampleInventory:
        if not self._config.layout.raw_images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self._config.layout.raw_images_dir}")
        if not self._config.layout.raw_labels_dir.exists():
            raise FileNotFoundError(f"Labels folder not found: {self._config.layout.raw_labels_dir}")

        image_paths = sorted(
            path
            for path in self._config.layout.raw_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self._config.image_extensions
        )
        label_paths = sorted(
            path for path in self._config.layout.raw_labels_dir.glob("*.txt") if path.is_file()
        )

        labels_by_stem = {path.stem: path for path in label_paths}
        images_by_stem = {path.stem: path for path in image_paths}

        matched_samples = [
            DatasetSample(image_path=image_path, label_path=labels_by_stem[image_path.stem])
            for image_path in image_paths
            if image_path.stem in labels_by_stem
        ]
        missing_labels = [
            image_path for image_path in image_paths if image_path.stem not in labels_by_stem
        ]
        missing_images = [
            label_path for label_path in label_paths if label_path.stem not in images_by_stem
        ]

        if not matched_samples:
            raise ValueError("No matched image/label pairs were found in data_raw.")

        return SampleInventory(
            matched_samples=matched_samples,
            missing_labels=missing_labels,
            missing_images=missing_images,
        )


class DatasetSplitter:
    def __init__(self, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
        self._seed = seed
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio

    def split(self, samples: list[DatasetSample]) -> DatasetSplit:
        ratio_sum = self._train_ratio + self._val_ratio + self._test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {ratio_sum}")

        shuffled = samples[:]
        random.Random(self._seed).shuffle(shuffled)

        train_count = int(len(shuffled) * self._train_ratio)
        val_count = int(len(shuffled) * self._val_ratio)
        test_count = len(shuffled) - train_count - val_count

        if min(train_count, val_count, test_count) <= 0:
            raise ValueError(
                "Invalid split. Need at least one sample in train, val, and test. "
                f"Got train={train_count}, val={val_count}, test={test_count} from {len(shuffled)} samples."
            )

        train_end = train_count
        val_end = train_count + val_count
        return DatasetSplit(
            train_samples=shuffled[:train_end],
            val_samples=shuffled[train_end:val_end],
            test_samples=shuffled[val_end:],
        )


class StageDatasetExporter:
    def __init__(self, datasets_root: Path) -> None:
        self._datasets_root = datasets_root

    def export(
        self,
        stage_config: StageConfig,
        split: DatasetSplit,
        class_catalog: SourceClassCatalog,
    ) -> StageExportSummary:
        stage_dir = self._datasets_root / stage_config.name
        stage_paths = self._reset_stage_dir(stage_dir)

        train_counts = self._write_split(
            samples=split.train_samples,
            image_dir=stage_paths["images_train"],
            label_dir=stage_paths["labels_train"],
            stage_config=stage_config,
            class_catalog=class_catalog,
        )
        val_counts = self._write_split(
            samples=split.val_samples,
            image_dir=stage_paths["images_val"],
            label_dir=stage_paths["labels_val"],
            stage_config=stage_config,
            class_catalog=class_catalog,
        )
        test_counts = self._write_split(
            samples=split.test_samples,
            image_dir=stage_paths["images_test"],
            label_dir=stage_paths["labels_test"],
            stage_config=stage_config,
            class_catalog=class_catalog,
        )

        self._write_yaml(stage_dir=stage_dir, class_names=stage_config.class_names)
        return StageExportSummary(
            output_dir=stage_dir,
            train_counts=train_counts,
            val_counts=val_counts,
            test_counts=test_counts,
        )

    def _write_split(
        self,
        samples: list[DatasetSample],
        image_dir: Path,
        label_dir: Path,
        stage_config: StageConfig,
        class_catalog: SourceClassCatalog,
    ) -> Counter:
        counts: Counter = Counter()
        written_image_count = 0
        written_label_count = 0

        for sample in samples:
            parsed_lines = self._parse_label_lines(sample.label_path, class_catalog)
            label_lines = self._build_stage_label_lines(parsed_lines, stage_config)

            self._write_stage_image(
                source_image_path=sample.image_path,
                target_image_path=image_dir / sample.image_path.name,
            )
            written_image_count += 1
            (label_dir / sample.label_path.name).write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            written_label_count += 1

            for line in label_lines:
                stage_id = int(line.split()[0])
                counts[stage_config.class_names[stage_id]] += 1

        self._validate_written_split(
            image_dir=image_dir,
            label_dir=label_dir,
            expected_images=written_image_count,
            expected_labels=written_label_count,
            counts=counts,
        )
        return counts

    def _parse_label_lines(
        self,
        label_path: Path,
        class_catalog: SourceClassCatalog,
    ) -> list[ParsedLabel]:
        parsed: list[ParsedLabel] = []

        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label format in {label_path} at line {line_number}: {raw_line!r}"
                )

            source_id = int(parts[0])
            class_name = class_catalog.name_for_id(source_id)
            x_center, y_center, width, height = map(float, parts[1:])
            self._validate_yolo_box(
                label_path=label_path,
                line_number=line_number,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
            parsed.append(
                ParsedLabel(
                    source_id=source_id,
                    class_name=class_name,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
            )

        return parsed

    def _build_stage_label_lines(
        self,
        parsed_lines: list[ParsedLabel],
        stage_config: StageConfig,
    ) -> list[str]:
        transformed: list[str] = []

        for parsed_label in parsed_lines:
            stage_id = stage_config.stage_id_for(parsed_label.class_name)
            transformed.append(f"{stage_id} {' '.join(parsed_label.yolo_coords)}")

        return transformed

    def _validate_yolo_box(
        self,
        label_path: Path,
        line_number: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ) -> None:
        values = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }
        for field_name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Invalid YOLO value in {label_path} at line {line_number}: "
                    f"{field_name}={value} is outside [0, 1]"
                )

        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"Invalid YOLO box in {label_path} at line {line_number}: "
                f"width={width}, height={height}, expected both > 0"
            )

        left = x_center - width / 2
        right = x_center + width / 2
        top = y_center - height / 2
        bottom = y_center + height / 2
        if left < 0.0 or right > 1.0 or top < 0.0 or bottom > 1.0:
            raise ValueError(
                f"Invalid YOLO box extent in {label_path} at line {line_number}: "
                f"box=({left}, {top}, {right}, {bottom}) must stay inside the image"
            )

    def _validate_written_split(
        self,
        image_dir: Path,
        label_dir: Path,
        expected_images: int,
        expected_labels: int,
        counts: Counter,
    ) -> None:
        actual_images = sorted(path for path in image_dir.iterdir() if path.is_file())
        actual_labels = sorted(path for path in label_dir.glob("*.txt") if path.is_file())
        if len(actual_images) != expected_images or len(actual_labels) != expected_labels:
            raise RuntimeError(
                f"Split write mismatch in {image_dir.parent}: "
                f"expected images={expected_images}, labels={expected_labels}, "
                f"got images={len(actual_images)}, labels={len(actual_labels)}"
            )

        image_stems = {path.stem for path in actual_images}
        label_stems = {path.stem for path in actual_labels}
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)
            missing_images = sorted(label_stems - image_stems)
            raise RuntimeError(
                f"Image/label stem mismatch in {image_dir.parent}: "
                f"missing_labels={missing_labels[:5]}, missing_images={missing_images[:5]}"
            )

        if actual_images and sum(counts.values()) <= 0:
            raise RuntimeError(
                f"No annotation boxes were written for split {image_dir.parent}. "
                "Refusing to export an empty detection target."
            )

    def _write_stage_image(self, source_image_path: Path, target_image_path: Path) -> None:
        with Image.open(source_image_path) as source_image:
            orientation = int((source_image.getexif() or {}).get(274, 1))
            if orientation == 1:
                shutil.copy2(source_image_path, target_image_path)
                return

            normalized = ImageOps.exif_transpose(source_image).convert("RGB")
            normalized.save(target_image_path)

    def _reset_stage_dir(self, stage_dir: Path) -> dict[str, Path]:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)

        paths = {
            "images_train": stage_dir / "images" / "train",
            "images_val": stage_dir / "images" / "val",
            "images_test": stage_dir / "images" / "test",
            "labels_train": stage_dir / "labels" / "train",
            "labels_val": stage_dir / "labels" / "val",
            "labels_test": stage_dir / "labels" / "test",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        return paths

    def _write_yaml(self, stage_dir: Path, class_names: list[str]) -> None:
        yaml_lines = [
            "path: .",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(class_names)}",
            "names:",
        ]
        for index, class_name in enumerate(class_names):
            yaml_lines.append(f"  {index}: {class_name}")

        (stage_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


class ClassificationDatasetExporter:
    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root

    def export(
        self,
        dataset_name: str,
        stage_config: StageConfig,
        split: DatasetSplit,
        class_catalog: SourceClassCatalog,
    ) -> ClassificationExportSummary:
        dataset_dir = self._output_root / dataset_name
        split_roots = self._reset_dataset_dir(dataset_dir, stage_config.class_names)

        train_counts = self._write_split(
            samples=split.train_samples,
            split_root=split_roots["train"],
            stage_config=stage_config,
            class_catalog=class_catalog,
        )
        val_counts = self._write_split(
            samples=split.val_samples,
            split_root=split_roots["val"],
            stage_config=stage_config,
            class_catalog=class_catalog,
        )
        test_counts = self._write_split(
            samples=split.test_samples,
            split_root=split_roots["test"],
            stage_config=stage_config,
            class_catalog=class_catalog,
        )
        return ClassificationExportSummary(
            output_dir=dataset_dir,
            train_counts=train_counts,
            val_counts=val_counts,
            test_counts=test_counts,
        )

    def _write_split(
        self,
        samples: list[DatasetSample],
        split_root: Path,
        stage_config: StageConfig,
        class_catalog: SourceClassCatalog,
    ) -> Counter:
        counts: Counter = Counter()
        for sample in samples:
            parsed_labels = self._parse_label_lines(sample.label_path, class_catalog)
            if not parsed_labels:
                continue

            with Image.open(sample.image_path) as source_image:
                # Normalize EXIF orientation so YOLO coordinates and pixel data share the same frame.
                image = ImageOps.exif_transpose(source_image).convert("RGB")
                image_width, image_height = image.size

                for object_index, parsed_label in enumerate(parsed_labels):
                    stage_id = stage_config.stage_id_for(parsed_label.class_name)
                    class_name = stage_config.class_names[stage_id]
                    left, top, right, bottom = self._to_square_pixel_box(
                        x_center=parsed_label.x_center,
                        y_center=parsed_label.y_center,
                        box_width=parsed_label.width,
                        box_height=parsed_label.height,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    if right <= left or bottom <= top:
                        continue

                    crop = image.crop((left, top, right, bottom))
                    image_key = f"{sample.image_path.stem}_{sample.image_path.suffix.lower().lstrip('.')}"
                    crop_name = f"{image_key}_{object_index}.png"
                    crop.save(split_root / class_name / crop_name)
                    counts[class_name] += 1

        if samples and sum(counts.values()) <= 0:
            raise RuntimeError(
                f"No crops were exported for split directory: {split_root}. "
                "Check source labels and YOLO boxes."
            )
        return counts

    def _parse_label_lines(
        self,
        label_path: Path,
        class_catalog: SourceClassCatalog,
    ) -> list[ParsedLabel]:
        parsed: list[ParsedLabel] = []
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label format in {label_path} at line {line_number}: {raw_line!r}"
                )

            source_id = int(parts[0])
            class_name = class_catalog.name_for_id(source_id)
            x_center, y_center, width, height = map(float, parts[1:])
            self._validate_yolo_box(
                label_path=label_path,
                line_number=line_number,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
            parsed.append(
                ParsedLabel(
                    source_id=source_id,
                    class_name=class_name,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
            )
        return parsed

    def _validate_yolo_box(
        self,
        label_path: Path,
        line_number: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ) -> None:
        values = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }
        for field_name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Invalid YOLO value in {label_path} at line {line_number}: "
                    f"{field_name}={value} is outside [0, 1]"
                )

        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"Invalid YOLO box in {label_path} at line {line_number}: "
                f"width={width}, height={height}, expected both > 0"
            )

        left = x_center - width / 2
        right = x_center + width / 2
        top = y_center - height / 2
        bottom = y_center + height / 2
        if left < 0.0 or right > 1.0 or top < 0.0 or bottom > 1.0:
            raise ValueError(
                f"Invalid YOLO box extent in {label_path} at line {line_number}: "
                f"box=({left}, {top}, {right}, {bottom}) must stay inside the image"
            )

    def _to_square_pixel_box(
        self,
        x_center: float,
        y_center: float,
        box_width: float,
        box_height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        side_pixels = int(round(max(box_width * image_width, box_height * image_height)))
        side_pixels = max(1, min(side_pixels, image_width, image_height))

        center_x_pixels = x_center * image_width
        center_y_pixels = y_center * image_height
        left = int(round(center_x_pixels - side_pixels / 2))
        top = int(round(center_y_pixels - side_pixels / 2))

        max_left = image_width - side_pixels
        max_top = image_height - side_pixels
        left = min(max(left, 0), max_left)
        top = min(max(top, 0), max_top)

        right = left + side_pixels
        bottom = top + side_pixels
        return left, top, right, bottom

    def _reset_dataset_dir(self, dataset_dir: Path, class_names: list[str]) -> dict[str, Path]:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        split_roots = {
            "train": dataset_dir / "train",
            "val": dataset_dir / "val",
            "test": dataset_dir / "test",
        }
        for split_root in split_roots.values():
            for class_name in class_names:
                (split_root / class_name).mkdir(parents=True, exist_ok=True)
        return split_roots


class DatasetPreparationPipeline:
    _STAGE_SPLIT_OFFSETS = {
        "stage1": 0,
        "stage2": 101,
        "stage3": 202,
    }

    def __init__(self, config: DatasetPreparationConfig) -> None:
        self._config = config
        self._scanner = SourceDatasetScanner(config)
        self._stage_exporter = StageDatasetExporter(datasets_root=config.layout.datasets_root)
        self._classification_exporter = ClassificationDatasetExporter(
            output_root=config.layout.classification_root
        )

    def run(self) -> DatasetPreparationReport:
        class_catalog = SourceClassCatalog.load(self._config.layout.raw_notes_path)
        inventory = self._scanner.gather_samples()
        stage_splits = self._build_stage_splits(inventory.matched_samples)

        self._cleanup_stale_outputs()
        self._config.layout.datasets_root.mkdir(parents=True, exist_ok=True)

        stage1_config, stage2_config, stage3_config = self._build_stage_configs(class_catalog)
        stage1_summary = self._stage_exporter.export(
            stage_config=stage1_config,
            split=stage_splits["stage1"],
            class_catalog=class_catalog,
        )
        detection_stage_summaries = {
            stage1_config.name: stage1_summary,
        }
        classification_summaries = {
            "stage2_material": self._classification_exporter.export(
                dataset_name="stage2_material",
                stage_config=stage2_config,
                split=stage_splits["stage2"],
                class_catalog=class_catalog,
            ),
            "stage3_denomination": self._classification_exporter.export(
                dataset_name="stage3_denomination",
                stage_config=stage3_config,
                split=stage_splits["stage3"],
                class_catalog=class_catalog,
            ),
        }

        return DatasetPreparationReport(
            seed=self._config.seed,
            train_ratio=self._config.train_ratio,
            val_ratio=self._config.val_ratio,
            test_ratio=self._config.test_ratio,
            shared_stage_split=self._config.shared_stage_split,
            stage_splits=stage_splits,
            inventory=inventory,
            stage_summaries=detection_stage_summaries,
            classification_summaries=classification_summaries,
        )

    def _build_stage_configs(self, class_catalog: SourceClassCatalog) -> list[StageConfig]:
        return [
            StageConfig(
                name="stage1",
                class_names=["coin"],
                name_to_stage_id={class_name: 0 for class_name in class_catalog.class_names},
            ),
            StageConfig(
                name="stage2",
                class_names=["bronze", "gold", "bicolor"],
                name_to_stage_id={
                    "1_cent": 0,
                    "2_cent": 0,
                    "5_cent": 0,
                    "10_cent": 1,
                    "20_cent": 1,
                    "50_cent": 1,
                    "1_euro": 2,
                    "2_euro": 2,
                },
            ),
            StageConfig(
                name="stage3",
                class_names=class_catalog.class_names,
                name_to_stage_id={
                    class_name: index for index, class_name in enumerate(class_catalog.class_names)
                },
            ),
        ]

    def _build_stage_splits(self, samples: list[DatasetSample]) -> dict[str, DatasetSplit]:
        if self._config.shared_stage_split:
            shared_split = self._split_with_seed(samples, self._config.seed)
            return {
                "stage1": shared_split,
                "stage2": shared_split,
                "stage3": shared_split,
            }

        return {
            stage_name: self._split_with_seed(
                samples,
                self._config.seed + self._STAGE_SPLIT_OFFSETS[stage_name],
            )
            for stage_name in ("stage1", "stage2", "stage3")
        }

    def _split_with_seed(self, samples: list[DatasetSample], seed: int) -> DatasetSplit:
        splitter = DatasetSplitter(
            seed=seed,
            train_ratio=self._config.train_ratio,
            val_ratio=self._config.val_ratio,
            test_ratio=self._config.test_ratio,
        )
        return splitter.split(samples)

    def _cleanup_stale_outputs(self) -> None:
        stale_dirs = [
            self._config.layout.datasets_root / "stage2",
            self._config.layout.datasets_root / "stage3",
        ]
        for stale_dir in stale_dirs:
            if stale_dir.exists():
                print(
                    f"Found stale dataset folder: {stale_dir}. "
                    "Removing before regeneration from data_raw..."
                )
                self._safe_rmtree(stale_dir)

    def _safe_rmtree(self, path: Path, retries: int = 4, retry_sleep_seconds: float = 0.4) -> None:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                shutil.rmtree(path)
                return
            except Exception as exc:  # pragma: no cover - platform specific
                last_error = exc
                if attempt == retries:
                    break
                print(
                    f"  Delete retry {attempt}/{retries - 1} for {path} "
                    f"(reason: {type(exc).__name__})."
                )
                time.sleep(retry_sleep_seconds)
        raise RuntimeError(f"Failed to remove directory after retries: {path}") from last_error


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = DatasetPreparationConfig.default()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare stage1 YOLO dataset and pre-cropped classification datasets "
            "(stage2_material/stage3_denomination)."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help=f"Random seed for the train/val/test split (default: {defaults.seed}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=defaults.train_ratio,
        help=f"Train split ratio (default: {defaults.train_ratio}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=defaults.val_ratio,
        help=f"Validation split ratio (default: {defaults.val_ratio}).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=defaults.test_ratio,
        help=f"Test split ratio (default: {defaults.test_ratio}).",
    )
    parser.add_argument(
        "--shared-stage-split",
        action="store_true",
        default=defaults.shared_stage_split,
        help=(
            "Reuse the same train/val/test image split for stage1/stage2_material/"
            "stage3_denomination. Default is False: each export gets its own deterministic split."
        ),
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> DatasetPreparationConfig:
    defaults = DatasetPreparationConfig.default()
    return DatasetPreparationConfig(
        layout=defaults.layout,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shared_stage_split=args.shared_stage_split,
        image_extensions=defaults.image_extensions,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    pipeline = DatasetPreparationPipeline(_config_from_args(args))
    report = pipeline.run()
    report.print_console()


if __name__ == "__main__":
    main()

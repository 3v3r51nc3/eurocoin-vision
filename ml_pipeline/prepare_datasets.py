from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class DatasetPreparationConfig:
    layout: ProjectLayout
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    image_extensions: frozenset[str]

    @classmethod
    def default(cls) -> "DatasetPreparationConfig":
        return cls(
            layout=ProjectLayout.from_current_file(),
            seed=52,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
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
class DatasetPreparationReport:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    split: DatasetSplit
    inventory: SampleInventory
    stage_summaries: dict[str, StageExportSummary]

    def print_console(self) -> None:
        print(f"Seed: {self.seed}")
        print(f"Train ratio: {self.train_ratio}")
        print(f"Val ratio: {self.val_ratio}")
        print(f"Test ratio: {self.test_ratio}")
        print(
            "Matched samples: "
            f"{len(self.split.train_samples) + len(self.split.val_samples) + len(self.split.test_samples)}"
        )
        print(f"Train samples: {len(self.split.train_samples)}")
        print(f"Val samples: {len(self.split.val_samples)}")
        print(f"Test samples: {len(self.split.test_samples)}")
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

            shutil.copy2(sample.image_path, image_dir / sample.image_path.name)
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
            f"path: {stage_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(class_names)}",
            "names:",
        ]
        for index, class_name in enumerate(class_names):
            yaml_lines.append(f"  {index}: {class_name}")

        (stage_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


class DatasetPreparationPipeline:
    def __init__(self, config: DatasetPreparationConfig) -> None:
        self._config = config
        self._scanner = SourceDatasetScanner(config)
        self._splitter = DatasetSplitter(
            seed=config.seed,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
        )
        self._exporter = StageDatasetExporter(datasets_root=config.layout.datasets_root)

    def run(self) -> DatasetPreparationReport:
        class_catalog = SourceClassCatalog.load(self._config.layout.raw_notes_path)
        inventory = self._scanner.gather_samples()
        split = self._splitter.split(inventory.matched_samples)

        self._config.layout.datasets_root.mkdir(parents=True, exist_ok=True)

        stage_summaries = {
            stage_config.name: self._exporter.export(
                stage_config=stage_config,
                split=split,
                class_catalog=class_catalog,
            )
            for stage_config in self._build_stage_configs(class_catalog)
        }

        return DatasetPreparationReport(
            seed=self._config.seed,
            train_ratio=self._config.train_ratio,
            val_ratio=self._config.val_ratio,
            test_ratio=self._config.test_ratio,
            split=split,
            inventory=inventory,
            stage_summaries=stage_summaries,
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
                    class_name: class_id for class_id, class_name in class_catalog.id_to_name.items()
                },
            ),
        ]


def main() -> None:
    pipeline = DatasetPreparationPipeline(DatasetPreparationConfig.default())
    report = pipeline.run()
    report.print_console()


if __name__ == "__main__":
    main()

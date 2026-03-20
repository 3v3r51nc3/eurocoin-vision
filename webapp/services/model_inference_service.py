from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import yaml

from webapp.config.app_config import AppConfig
from webapp.models.inference_settings import InferenceSettings
from webapp.models.pipeline_prediction import CoinDetection, PipelinePrediction

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(slots=True)
class ClassifierCheckpoint:
    model: nn.Module
    class_names: list[str]
    image_size: int
    device: torch.device
    _transform: transforms.Compose = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Return the predicted class name and its softmax probability."""
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        predicted_index = int(probabilities.argmax().item())
        return self.class_names[predicted_index], float(probabilities[predicted_index].item())


class EuroCoinPipeline:
    def __init__(
        self,
        detector: YOLO,
        material_classifier: ClassifierCheckpoint,
        denomination_classifiers: dict[str, ClassifierCheckpoint],
    ) -> None:
        self._detector = detector
        self._material_classifier = material_classifier
        self._denomination_classifiers = denomination_classifiers

    def predict(
        self,
        image_array: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
        padding_ratio: float = 0.05,
    ) -> PipelinePrediction:
        """Run the three-stage detection and classification pipeline on an image array."""
        image = Image.fromarray(image_array).convert("RGB")
        result = self._detector.predict(
            source=np.array(image),
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False,
        )[0]

        detections: list[CoinDetection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return PipelinePrediction(input_image=image, detections=detections)

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        width, height = image.size

        for box_xyxy, confidence in zip(boxes, confidences):
            left, top, right, bottom = self._expand_box(
                box_xyxy=box_xyxy,
                image_width=width,
                image_height=height,
                padding_ratio=padding_ratio,
            )

            crop = image.crop((left, top, right, bottom))
            material_name, _ = self._material_classifier.predict(crop)
            denomination_name, _ = self._denomination_classifiers[material_name].predict(crop)
            detections.append(
                CoinDetection(
                    box_xyxy=(left, top, right, bottom),
                    detection_confidence=float(confidence),
                    material=material_name,
                    denomination=denomination_name,
                )
            )

        detections.sort(key=lambda detection: (detection.box_xyxy[1], detection.box_xyxy[0]))
        return PipelinePrediction(input_image=image, detections=detections)

    def _expand_box(
        self,
        box_xyxy: np.ndarray,
        image_width: int,
        image_height: int,
        padding_ratio: float,
    ) -> tuple[int, int, int, int]:
        """Expand a bounding box by a fractional padding and clamp to image bounds."""
        left, top, right, bottom = [float(value) for value in box_xyxy]
        box_width = right - left
        box_height = bottom - top
        pad_x = box_width * padding_ratio
        pad_y = box_height * padding_ratio

        expanded_left = max(0, int(round(left - pad_x)))
        expanded_top = max(0, int(round(top - pad_y)))
        expanded_right = min(image_width, int(round(right + pad_x)))
        expanded_bottom = min(image_height, int(round(bottom + pad_y)))
        return expanded_left, expanded_top, expanded_right, expanded_bottom


class ModelInferenceService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_pipeline(base_dir: str) -> EuroCoinPipeline:
        """Discover and load all model weights from model_weights/, cached across sessions."""
        base_path = Path(base_dir)
        model_weights_dir = base_path / "model_weights"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        stage1_weights_path = ModelInferenceService._find_latest_file(model_weights_dir / "stage1", "*.pt")
        stage2_weights_path = ModelInferenceService._find_latest_file(model_weights_dir / "stage2", "*.pt")
        stage3_manifest_path = ModelInferenceService._find_latest_file(model_weights_dir / "stage3", "*.yaml")

        stage3_manifest = yaml.safe_load(stage3_manifest_path.read_text(encoding="utf-8"))
        material_names = list(stage3_manifest["mapping"].keys())
        stage3_weights = {
            material_name: ModelInferenceService._find_latest_file(
                model_weights_dir / f"stage3_{material_name}",
                "*.pt",
            )
            for material_name in material_names
        }

        detector = YOLO(str(stage1_weights_path))
        material_classifier = ModelInferenceService._load_classifier_checkpoint(stage2_weights_path, device)
        denomination_classifiers = {
            material_name: ModelInferenceService._load_classifier_checkpoint(weights_path, device)
            for material_name, weights_path in stage3_weights.items()
        }
        return EuroCoinPipeline(
            detector=detector,
            material_classifier=material_classifier,
            denomination_classifiers=denomination_classifiers,
        )

    @staticmethod
    def _find_latest_file(root_dir: Path, pattern: str) -> Path:
        """Return the most recently modified file matching pattern under root_dir."""
        if not root_dir.exists():
            raise FileNotFoundError(f"Missing directory: {root_dir}")

        matches = sorted(
            (path for path in root_dir.rglob(pattern) if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            raise FileNotFoundError(f"No files matching {pattern!r} under {root_dir}")
        return matches[0]

    @staticmethod
    def _load_classifier_checkpoint(checkpoint_path: Path, device: torch.device) -> ClassifierCheckpoint:
        """Load a ResNet18 checkpoint and return a ready-to-use ClassifierCheckpoint."""
        checkpoint = ModelInferenceService._torch_load(checkpoint_path, device)
        class_names = list(checkpoint["class_names"])
        image_size = int(checkpoint.get("image_size", 224))

        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return ClassifierCheckpoint(
            model=model,
            class_names=class_names,
            image_size=image_size,
            device=device,
        )

    @staticmethod
    def _torch_load(checkpoint_path: Path, device: torch.device) -> dict:
        """Load a checkpoint file, adapting to torch.load API changes across versions."""
        load_kwargs: dict[str, object] = {"map_location": device}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        return torch.load(checkpoint_path, **load_kwargs)

    def predict(self, image_array: np.ndarray, settings: InferenceSettings) -> PipelinePrediction:
        """Run inference using the cached pipeline and the given inference settings."""
        pipeline = self._load_pipeline(str(self._config.base_dir))
        return pipeline.predict(
            image_array=image_array,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
        )

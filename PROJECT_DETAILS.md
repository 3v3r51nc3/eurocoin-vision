# eurocoin-vision - Technical Details

This document contains the full technical breakdown of the project (pipeline design, dataset engineering, training, evaluation, metrics, and repository internals).

If you want the concise overview first, start from [README.md](README.md).

`eurocoin-vision` is my personal deep learning reinterpretation of **[Image_projet_money](https://github.com/sorooshaghaei/Image_projet_money)** - a group academic project developed during the **M1 Vision et Machine Intelligente** program at **Université Paris Cité**.

The objective is unchanged: given a photo containing euro coins, detect every coin and identify its denomination. The difference is the engineering approach. Instead of hand-crafted image processing, this repository builds the solution as a staged machine learning pipeline using YOLOv8 for detection and ResNet18 classifiers for material and denomination recognition.

This project demonstrates:

- multi-stage ML pipeline design
- dataset transformation and validation
- object detection and image classification workflows
- hierarchical label decomposition
- experiment artifact publishing
- lightweight productization through a Streamlit inference app

## 1. Project Overview

The problem is small in scope but technically meaningful: euro coins are visually similar, can appear in cluttered scenes, and must be counted and classified reliably at the instance level.

Deep learning was chosen here to replace a classical CV approach with a pipeline that can learn appearance directly from annotated data and scale more naturally to variation in lighting, pose, background, and partial overlap.

The system implemented in this repository performs:

- coin detection in the full image
- crop extraction for each detected coin
- material prediction: `bronze`, `gold`, or `bicolor`
- denomination prediction within the corresponding material family
- browser-based visualization and result summarization through Streamlit

## 2. Example Task

**Input**

An image containing a mixed set of euro coins.

**Output**

- the number of detected coins
- the predicted denomination of each coin

In practice, the task is harder than simple counting because the pipeline must handle:

- overlapping or closely packed coins
- similar color distributions across denominations
- lighting variation and reflections on metal surfaces
- scale changes and background clutter

## 3. Pipeline Architecture

The repository implements a three-stage pipeline:

- **Stage 1 - Coin Detection**: a YOLOv8 detector localizes all coins in the image as a single class, `coin`
- **Stage 2 - Material Classification**: a ResNet18 classifier predicts the coin material group: `bronze`, `gold`, or `bicolor`
- **Stage 3 - Denomination Classification**: a material-specific ResNet18 classifier predicts the final denomination inside that material family

This decomposition is intentional. Instead of asking one model to solve full localization plus 8-way denomination discrimination at once, the pipeline separates:

- spatial localization
- coarse semantic grouping
- fine-grained denomination recognition

That makes the system easier to debug, easier to retrain per stage, and better aligned with the structure of euro coin appearance.

```text
Input Image
     |
     v
Stage 1 - YOLOv8 Coin Detection
     |
     v
Detected coin boxes
     |
     v
Crop each detected coin
     |
     v
Stage 2 - ResNet18 Material Classification
(bronze / gold / bicolor)
     |
     v
Stage 3 - Material-Specific ResNet18 Denomination Classification
(1c, 2c, 5c) or (10c, 20c, 50c) or (1EUR, 2EUR)
     |
     v
Per-coin denomination predictions
```

### Data Flow Between Stages

- During dataset preparation, the raw YOLO annotations are exported into three detection datasets with the same image split.
- During training, `DetectionCropDatasetBuilder` converts Stage 2 and Stage 3 detection labels into cropped `ImageFolder` classification datasets.
- During inference, Stage 1 predicts boxes on the full image.
- Each predicted box is expanded slightly, cropped, sent to the Stage 2 classifier, then routed to the matching Stage 3 classifier.
- The Streamlit app aggregates the per-coin predictions into count, denomination summary, and estimated total value.

## 4. Tech Stack

- **Python**: core language for data preparation, training, inference, and the app
- **Ultralytics YOLOv8**: Stage 1 coin detector
- **PyTorch**: training and checkpointing for the classification models
- **Torchvision ResNet18**: Stage 2 and Stage 3 classifiers, initialized from ImageNet weights during training
- **Pillow + NumPy**: image loading, crop generation, and annotation rendering
- **PyYAML**: dataset configs and Stage 3 manifest serialization
- **Streamlit**: interactive inference UI with upload, threshold controls, annotated outputs, and result tables

OpenCV is not currently used in this repository. Image I/O and visualization are handled with Pillow, NumPy, and Streamlit instead.

## 5. Dataset Engineering

`ml_pipeline/data_raw/` is the source of truth for the project.

Dataset source: [Hugging Face - 3v3r51nc3/eurocoin-vision-dataset](https://huggingface.co/datasets/3v3r51nc3/eurocoin-vision-dataset)

The image collection was produced collaboratively by students from the **M1 Vision et Machine Intelligente** program at **Université Paris Cité (2026)** through real-world euro coin photographs captured under varying conditions.  
The full raw dataset annotation was carried out manually by the project author, including bounding boxes and denomination labels for each visible coin. This annotation work was a key part of the project, since the downstream multi-stage pipeline depends on consistent labeling across detection, material grouping, and denomination recognition tasks.

The raw dataset contains:

- `images/`: source photos
- `labels/`: YOLO-format annotations
- `notes.json`: the class taxonomy used to map integer ids to denomination names

Current raw dataset size discovered in the repository:

- `96` matched images
- `96` matched label files
- `917` annotated coin instances

Raw denomination counts:

| Class | Instances |
| --- | ---: |
| `1_cent` | 92 |
| `2_cent` | 141 |
| `5_cent` | 99 |
| `10_cent` | 135 |
| `20_cent` | 114 |
| `50_cent` | 134 |
| `1_euro` | 117 |
| `2_euro` | 85 |

### Stage Datasets Generated by `prepare_datasets.py`

`ml_pipeline/prepare_datasets.py` scans the raw dataset, validates image/label integrity, checks YOLO box bounds, applies a deterministic split with seed `52`, and exports three derived YOLO datasets under `ml_pipeline/datasets/`.

The split is shared across all stages:

- train: `67` images
- val: `14` images
- test: `15` images

#### Stage 1 dataset

Path: `ml_pipeline/datasets/stage1`

Classes:

- `coin`

Purpose:

- teach the detector to localize coins without denomination complexity

#### Stage 2 dataset

Path: `ml_pipeline/datasets/stage2`

Classes:

- `bronze`
- `gold`
- `bicolor`

Mapping implemented in code:

- `1_cent`, `2_cent`, `5_cent` -> `bronze`
- `10_cent`, `20_cent`, `50_cent` -> `gold`
- `1_euro`, `2_euro` -> `bicolor`

Purpose:

- learn a coarse visual partition before fine-grained denomination prediction

#### Stage 3 dataset

Path: `ml_pipeline/datasets/stage3`

Classes:

- original euro coin denominations from `notes.json`

Purpose:

- preserve denomination-level labels for final recognition

### Why Three Datasets?

The repository splits one raw annotation source into three supervised tasks because the tasks have different learning objectives:

- Stage 1 focuses only on localization
- Stage 2 focuses on coarse material appearance
- Stage 3 focuses on denomination discrimination

For Stage 2 and Stage 3 training, the notebook creates cropped classification datasets from the detection labels:

- `ml_pipeline/classification_datasets/stage2_material`
- `ml_pipeline/classification_datasets/stage3_denomination`
- `ml_pipeline/classification_datasets/stage3_by_material`

This is an important engineering detail: the project reuses one annotation source to support both detection and classification training.

## 6. Training

Training is orchestrated from `ml_pipeline/train_model.ipynb`.

The notebook contains:

- project root discovery for local environments and Google Colab
- optional Google Drive mounting
- dataset preparation invocation via `prepare_datasets.py`
- YOLO training for Stage 1
- crop dataset generation for Stages 2 and 3
- ResNet18 training and checkpoint publishing for classification stages

### Stage 1 - YOLOv8 Detector

Defined trainer: `Stage1YoloTrainer`

Current source configuration:

- base model: `yolov8n.pt`
- image size: `640`
- batch size: `32`
- epochs: `64`
- workers: `2`

The notebook source currently defines `64` epochs in the stage run cells, and this value can be adjusted per run.

Published artifact:

- `model_weights/stage1/stage1_yolov8n_best.pt`

### Stage 2 - Material Classifier

Defined trainer: `Stage2ResNetTrainer`

Model details:

- architecture: `ResNet18`
- initialization: `ResNet18_Weights.DEFAULT`
- loss: cross-entropy
- optimizer: `AdamW`
- image size: `224`
- learning rate: `1e-3`

Training data is produced by cropping every annotated Stage 2 box into an `ImageFolder` dataset.

Current source configuration:
- epochs: `64`
- batch size: `32`

Current Stage 2 crop split sizes (from `ml_pipeline/datasets/stage2` labels):

- train samples: `653`
- val samples: `96`
- test samples: `168`

Published artifact:

- `model_weights/stage2/stage2_resnet18_checkpoint.pt`

### Stage 3 - Hierarchical Denomination Classifiers

Defined trainer: `Stage3HierarchicalTrainer`

Current source configuration:
- epochs: `64`
- batch size: `32`

Instead of one 8-class classifier, Stage 3 trains three material-specific ResNet18 models:

- `stage3_bronze` -> `1_cent`, `2_cent`, `5_cent`
- `stage3_gold` -> `10_cent`, `20_cent`, `50_cent`
- `stage3_bicolor` -> `1_euro`, `2_euro`

The notebook exports a manifest at:

- `model_weights/stage3/stage3_resnet18_hierarchical.yaml`

Published classifier artifacts:

- `model_weights/stage3_bronze/stage3_resnet18_bronze_checkpoint.pt`
- `model_weights/stage3_gold/stage3_resnet18_gold_checkpoint.pt`
- `model_weights/stage3_bicolor/stage3_resnet18_bicolor_checkpoint.pt`

Recorded training set sizes from the repository:

| Stage 3 subset | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Bronze | 243 | 37 | 52 |
| Gold | 267 | 39 | 77 |
| Bicolor | 143 | 20 | 39 |

### Classification Augmentation

The classification pipeline applies augmentation during training:

- resize to `224 x 224`
- random rotation up to `360` degrees
- random vertical flip
- random affine translation and scale
- color jitter
- ImageNet normalization

## 7. Evaluation

Evaluation is implemented in `ml_pipeline/evaluate_model.ipynb`, which loads published checkpoints from `model_weights/` and reports:

- Stage 1 detector quality (`YOLO.val`)
- classifier quality on ground-truth boxes (to isolate classification from localization)
- full pipeline quality (detection + classification together) with IoU matching at `0.5`

Latest report:

- `ml_pipeline/results/evaluation_report_20260315_160709.json`
- created at: `2026-03-15T16:07:09`
- device: `CPU` (`torch-2.10.0+cpu`)
- evaluated split: `test` (`15` images, `164` coin instances)

Root cause of the previous quality drop (now fixed):

- a subset of dataset annotations was incorrect
- some dataset images had EXIF rotation/orientation mismatches
- this created label/pixel misalignment, which propagated errors through all pipeline stages

### 7.1 Stage Metrics

| Component | Metric | Value |
| --- | --- | ---: |
| Stage 1 detector | precision | `0.9937` |
| Stage 1 detector | recall | `1.0000` |
| Stage 1 detector | mAP50 | `0.9950` |
| Stage 1 detector | mAP50-95 | `0.9251` |
| Stage 2 on GT boxes | material accuracy | `0.9219` |
| Stage 3 on GT boxes (oracle material) | denomination accuracy | `0.8659` |
| Stage 3 on GT boxes (hierarchical) | denomination accuracy | `0.8232` |

Per-material Stage 3 hierarchical accuracy on GT boxes:

| Material | Accuracy |
| --- | ---: |
| `bicolor` | `0.8000` |
| `gold` | `0.8485` |
| `bronze` | `0.8095` |

### 7.2 Full Pipeline Metrics

| Area | Metric | Value |
| --- | --- | ---: |
| Detection | precision | `0.8902` |
| Detection | recall | `0.8902` |
| Detection | F1 | `0.8902` |
| Detection | mean IoU (matched) | `0.9419` |
| Detection | TP / FP / FN | `146 / 18 / 18` |
| Classification on matched detections | Stage 2 material accuracy | `0.9589` |
| Classification on matched detections | Stage 3 denomination accuracy | `0.8836` |
| End-to-end | precision | `0.7866` |
| End-to-end | recall | `0.7866` |
| End-to-end | F1 | `0.7866` |
| End-to-end | exact image match rate | `0.3333` |
| End-to-end | TP / FP / FN | `129 / 35 / 35` |

Per-class end-to-end recall:

| Class | Recall |
| --- | ---: |
| `1_cent` | `0.6429` |
| `2_cent` | `0.8846` |
| `5_cent` | `0.6957` |
| `10_cent` | `0.8929` |
| `20_cent` | `0.8421` |
| `50_cent` | `0.6842` |
| `1_euro` | `0.7368` |
| `2_euro` | `0.8125` |

## 8. Known Limitations and Priority Work

- Lowest denomination recalls are still `1_cent`, `5_cent`, and `50_cent`.

  Evidence: end-to-end recalls are `0.6429`, `0.6957`, and `0.6842`, respectively. These classes remain priority targets for hard-example mining and more balanced coverage.

- End-to-end quality still drops because of classification errors after correct detection.

  Evidence: detector finds `146` true matches, but only `129` become fully correct denomination predictions. That `17`-coin gap still contributes to final `FP/FN` (`35/35`) and limits exact-image success to `33.33%`.

- Hierarchical routing still introduces additional error on top of denomination classification.

  Evidence: Stage 3 denomination accuracy decreases from `0.8659` (oracle material) to `0.8232` (predicted material), indicating residual error propagation from Stage 2 to Stage 3.

## 9. Repository Structure

```text
eurocoin-vision/
├── main.py
├── README.md
├── requirements.txt
├── model_weights/
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   ├── stage3_bicolor/
│   ├── stage3_bronze/
│   └── stage3_gold/
├── ml_pipeline/
│   ├── data_raw/
│   │   ├── images/
│   │   ├── labels/
│   │   └── notes.json
│   ├── datasets/
│   │   ├── stage1/
│   │   ├── stage2/
│   │   └── stage3/
│   ├── results/
│   │   ├── evaluation_report_YYYYMMDD_HHMMSS.json
│   ├── prepare_datasets.py
│   ├── train_model.ipynb
│   ├── evaluate_model.ipynb
└── webapp/
    ├── app.py
    ├── config/
    ├── models/
    ├── services/
    └── views/
```

### Directory Roles

- `ml_pipeline/data_raw/`: source annotations and images
- `ml_pipeline/datasets/`: generated YOLO datasets for the three stages
- `ml_pipeline/prepare_datasets.py`: dataset validation, splitting, relabeling, and `data.yaml` export
- `ml_pipeline/train_model.ipynb`: end-to-end training workflow
- `ml_pipeline/evaluate_model.ipynb`: end-to-end evaluation workflow
- `ml_pipeline/results/evaluation_report_*.json`: saved structured reports from the evaluation notebook
- `model_weights/`: published detector and classifier checkpoints used by the app
- `webapp/`: Streamlit inference application
- `main.py`: app entrypoint, intended to be launched with `streamlit run main.py`

## 10. Future Work

- expand bronze and low-value coin data (`1_cent`, `2_cent`, `5_cent`) with higher intra-class diversity
- rebalance Stage 3 training with class-aware sampling and/or weighted loss for bronze subsets
- improve Stage 2 routing robustness to reduce oracle-vs-hierarchical accuracy gap
- tune Stage 1 inference settings and retraining strategy to reduce residual misses (`18 FN`) and extra detections (`18 FP`)
- keep reporting standardized end-to-end metrics from `evaluate_model.ipynb` for every model update
- optimize inference throughput for near real-time usage on CPU

## Usage Notes

Runtime and reproducibility instructions (dataset download, model weights, quick reproduce steps, and demo launch) are intentionally maintained in the concise overview:

- [README.md](README.md)

---

Back to concise overview: [README.md](README.md)

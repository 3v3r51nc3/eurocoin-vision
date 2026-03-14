# eurocoin-vision

`eurocoin-vision` is a deep learning reworking of my earlier classical computer vision project, `Image_projet_money`.

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

Dataset source: LATER

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

- train: `76` images
- val: `9` images
- test: `11` images

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
- batch size: `16`
- workers: `2`

The notebook source currently defines `18` epochs in the end-to-end pipeline cell, while the recorded experiment stored in `ml_pipeline/results/train_model_with_results.json` used `8` epochs for the published Stage 1 run.

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

Current source defaults for the classifier trainer:

- epochs: `5`
- batch size: `32`

Recorded experiment in `train_model_with_results.json`:

- epochs: `24`
- train samples: `701`
- val samples: `111`
- test samples: `105`

Published artifact:

- `model_weights/stage2/stage2_resnet18_checkpoint.pt`

### Stage 3 - Hierarchical Denomination Classifiers

Defined trainer: `Stage3HierarchicalTrainer`

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
| Bronze | 257 | 47 | 28 |
| Gold | 288 | 42 | 53 |
| Bicolor | 156 | 22 | 24 |

### Classification Augmentation

The classification pipeline applies augmentation during training:

- resize to `224 x 224`
- random rotation up to `360` degrees
- random vertical flip
- random affine translation and scale
- color jitter
- ImageNet normalization

## 7. Evaluation

Evaluation is implemented inside the training notebook:

- Stage 1 uses `YOLO(...).val(...)` on the exported test split
- Stage 2 and Stage 3 evaluate with classification loss and accuracy on held-out test crops

Recorded metrics from `ml_pipeline/results/train_model_with_results.json`:

| Stage | Model | Task | Test metrics |
| --- | --- | --- | --- |
| Stage 1 | YOLOv8n | coin detection | precision `1.000`, recall `0.546`, mAP50 `0.905`, mAP50-95 `0.832` |
| Stage 2 | ResNet18 | material classification | test loss `0.3777`, test accuracy `0.8667` |
| Stage 3 bronze | ResNet18 | `1c` vs `2c` vs `5c` | test loss `0.9571`, test accuracy `0.6071` |
| Stage 3 gold | ResNet18 | `10c` vs `20c` vs `50c` | test loss `0.5640`, test accuracy `0.7925` |
| Stage 3 bicolor | ResNet18 | `1EUR` vs `2EUR` | test loss `0.2138`, test accuracy `0.9167` |

Two points are important when reading these numbers:

- the repository currently records per-stage metrics rather than a full end-to-end pipeline benchmark
- Stage 1 recall is the main bottleneck, because missed detections cannot be recovered by later classifiers

## 8. Known Limitations

- `TODO: Fix poor Stage-1 recall`

  Stage 1 currently achieves strong precision but substantially lower recall than is desirable for a counting system. This means some coins are not detected at all, and every missed detection removes that coin from the rest of the pipeline.

- Bronze coin recognition remains the weakest denomination subset.

  The dataset currently contains significantly fewer bronze coins than other categories, which limits the model's ability to learn robust bronze representations. Increasing the number and diversity of bronze coin samples is expected to improve performance.

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
│   │   └── train_model_with_results.json
│   ├── prepare_datasets.py
│   ├── train_model.ipynb
│   └── demonstrate_model.ipynb
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
- `ml_pipeline/demonstrate_model.ipynb`: notebook-based inference demo using published weights
- `ml_pipeline/results/train_model_with_results.json`: stored notebook run with training logs and evaluation metrics
- `model_weights/`: published detector and classifier checkpoints used by the app
- `webapp/`: Streamlit inference application
- `main.py`: app entrypoint, intended to be launched with `streamlit run main.py`

## 10. Future Work

- expand the dataset, especially for underrepresented and difficult bronze examples
- improve Stage 1 recall to reduce missed coins
- strengthen augmentation and hard-example coverage for denomination classification
- improve crop quality and coin separation for crowded scenes
- add end-to-end pipeline evaluation, not only stage-wise metrics
- optimize the system for faster or real-time inference

## Running the Demo

Install dependencies and start the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run main.py
```

The app loads the latest published weights from `model_weights/`, lets the user set confidence and IoU thresholds, and returns:

- the uploaded image
- an annotated prediction image
- detected coin count
- estimated total value
- denomination summary and raw detection tables

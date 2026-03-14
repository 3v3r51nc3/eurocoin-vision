# eurocoin-vision

`eurocoin-vision` is a reworking of my previous project `Image_projet_money`.

The final task remains the same: determine how many coins are present in a photo and identify the denomination of each coin.

The main difference is the approach. In `Image_projet_money`, the task was solved with classical computer vision. In this project, a completely different approach is used for the same assignment: `Deep Learning`, specifically `Ultralytics YOLOv8`.

## Stack

- `Python`
- `Ultralytics YOLOv8`
- `ResNet`
- `Streamlit`

## Status

The project is still in progress.

At the moment the repository mainly contains:

- the master raw dataset in `ml_pipeline/data_raw/`
- dataset preparation scripts
- a training notebook

## Dataset

Source dataset: LATER

`ml_pipeline/data_raw/` is the source of truth for annotations. From it, the preparation script generates three YOLO datasets with the same train/val split:

- `ml_pipeline/datasets/stage1`: one class, all coins merged into `coin`
- `ml_pipeline/datasets/stage2`: three classes, `bronze`, `gold`, `bicolor`
- `ml_pipeline/datasets/stage3`: original denomination classes

## Quick Start

Soon
# eurocoin-vision

`eurocoin-vision` is a reworking of my previous project `Image_projet_money`.

The final task remains the same: determine how many coins are present in a photo and identify the denomination of each coin.

The main difference is the approach. In `Image_projet_money`, the task was solved with classical computer vision. In this project, a completely different approach is used for the same assignment: `Deep Learning`, specifically `Ultralytics YOLOv8`.

## Stack

- `Python`
- `Ultralytics YOLOv8`
- `Streamlit`

## Project Status

The project is still under active development and is not finished yet.

At the moment, the repository already includes:

- raw images and annotations
- a dataset preparation script for YOLO format
- a ready `train` / `val` split
- a notebook for model training
- dependencies for the future `Streamlit` interface

## Dataset

Source dataset: [Euro Coins Dataset on Kaggle](https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset)

## Project Goal

The goal is to build a system that can:

- detect coins in an image
- identify the denomination of each detected coin
- count the total number of coins in the photo

Current dataset classes:

- `1_cent`
- `2_cent`
- `5_cent`
- `10_cent`
- `20_cent`
- `50_cent`
- `1_euro`
- `2_euro`

## Repository Structure

```text
eurocoin-vision/
├── data_raw/              # raw images and annotations
├── dataset/               # dataset in YOLO format
│   ├── images/train
│   ├── images/val
│   ├── labels/train
│   ├── labels/val
│   └── data.yaml
├── prepare_dataset.py     # dataset preparation and split
├── train_model.ipynb      # YOLOv8 training
└── requirements.txt       # project dependencies
```

## Current Dataset State

- raw images: `150`
- training images: `120`
- validation images: `30`
- classes: `8`

The split is generated automatically in [prepare_dataset.py](prepare_dataset.py) with an `80/20` ratio.

## How It Works Right Now

1. Raw data is stored in `data_raw/images` and `data_raw/labels`.
2. [prepare_dataset.py](prepare_dataset.py) creates the YOLO dataset structure and generates `dataset/data.yaml`.
3. Model training is currently done in [train_model.ipynb](train_model.ipynb).
4. The next step is adding inference and the `Streamlit` interface.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python prepare_dataset.py
```

Model training is currently done through the notebook:

```text
train_model.ipynb
```

## Why This Is a Separate Project

This is not just a copy of `Image_projet_money`, but a new implementation of the same task.

Instead of a manually engineered classical computer vision pipeline, this version uses object detection with `YOLOv8`. This approach is closer to modern computer vision workflows and is better suited for scenes where multiple coins appear in the same image.

## Planned Next Steps

- finalize the training workflow outside the notebook or keep the notebook as the stable training entry point
- add inference for local images
- build a complete `Streamlit` application
- display detections, coin counts, and denominations in the UI

## Summary

`eurocoin-vision` is a new DL-based version of `Image_projet_money`. It solves the same task, but with a completely different approach: `Ultralytics YOLOv8` instead of classical computer vision. The final goal is simple: given a photo, determine how many euro coins are present and identify the denomination of each one.

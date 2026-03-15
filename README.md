# eurocoin-vision

`eurocoin-vision` is a deep learning computer vision project that detects euro coins in an image, counts them, and predicts each denomination.

This repository is my personal DL re-implementation of the classical CV project [Image_projet_money](https://github.com/sorooshaghaei/Image_projet_money), developed during M1 Vision et Machine Intelligente (Universite Paris Cite).

## Quick Portfolio Summary

- Problem: instance-level euro coin detection + denomination recognition
- Architecture: 3-stage pipeline (`YOLOv8` detector + `ResNet18` material classifier + material-specific `ResNet18` denomination classifiers)
- Scope: end-to-end ML pipeline (data prep, training, evaluation, inference app)
- Demo: Streamlit web app

## Key Results (latest test report)

Source: `ml_pipeline/results/evaluation_report_20260314_234921.json`

- Stage 1 detection mAP50: `0.9924`
- Stage 1 detection mAP50-95: `0.9254`
- Stage 2 material accuracy (GT crops): `0.9048`
- Stage 3 denomination accuracy (hierarchical, GT crops): `0.7083`
- End-to-end denomination F1: `0.6787`

## Visual Results (placeholders)

Use this section to add real project screenshots.  
Suggested folder: `assets/screenshots/`

### 1. End-to-end app example
- Streamlit page with both original image and annotated prediction visible.
- Metrics visible (`Detected coins`, `Estimated total value`, `Unique denominations`).

![End-to-end app example](assets/screenshots/01_end_to_end_app.png)

### 2. Good prediction case

![Good prediction case](assets/screenshots/02_good_case.png)

### 3. Hard case
- Difficult image (overlap, reflections, small bronze coins, clutter).

![Hard case](assets/screenshots/03_hard_case.png)
![Hard case table](assets/screenshots/03_hard_case_table.png)

### 4. Evaluation artifact
![Evaluation artifact](assets/screenshots/04_evaluation_metrics.png)

## Tech Stack

- Python
- Ultralytics YOLOv8
- PyTorch + Torchvision (ResNet18)
- Pillow + NumPy
- PyYAML
- Streamlit

## Data and Weights

- Dataset: [Hugging Face - 3v3r51nc3/eurocoin-vision-dataset](https://huggingface.co/datasets/3v3r51nc3/eurocoin-vision-dataset)
- Pretrained weights: [GitHub Releases](https://github.com/3v3r51nc3/eurocoin-vision/releases/)

## Quick Reproduce

1. Download dataset from Hugging Face and place contents under `ml_pipeline/data_raw/` (`images/`, `labels/`, `notes.json`).
2. Download weights from GitHub Releases and place them under `model_weights/` (`stage1/`, `stage2/`, `stage3/`, `stage3_bronze/`, `stage3_gold/`, `stage3_bicolor/`).
3. Prepare datasets:

```bash
python ml_pipeline/prepare_datasets.py
```

4. Run the app:

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Project Details

For full technical documentation (pipeline architecture, dataset engineering, training configs, evaluation protocol, per-class metrics, known limitations, and repository structure), see:

- [PROJECT_DETAILS.md](PROJECT_DETAILS.md)

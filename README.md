# Deep-learning-wound-segmentation
# Wound Segmentation Using WSNet [Paper Reproduction]

This project reproduces and extends the "WSNet" architecture for wound image segmentation [WACV 2023], using a local-global combined approach. 

## Highlights
- **Achieved Dice Score:** 0.805 (combined model, FPN backbone)
- **Tech Stack:** PyTorch, segmentation-models-pytorch, albumentations
- **Approach:** Combined global image context and local patch detail, following WSNet principles.

## Directory Structure

See above.

## How to Run

1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. Prepare data (see `/data/README.md`).
3. Train models:
    ```
    python src/FPN_global_model_training_code.py
    python src/FPN_local_model_training_code.py
    python src/FPN_combined_model_training_code.py
    ```
4. Review logs and generated models in `/logs`.

See the report for more details on the experimental setup and results.

## Dataset

No data included. Please contact (your_email@domain) or see [instructions](data/README.md) for dataset info and format.

## Reference

Oota et al., "WSNet: Towards An Effective Method for Wound Image Segmentation," WACV 2023.

# Deep-learning-wound-segmentation

Wound Segmentation Using WSNet [Paper Reproduction]

This project reproduces and extends the "WSNet" architecture for wound image segmentation [WACV 2023], using a local-global combined approach. 

## Highlights

- **State-of-the-art Global-Local Segmentation:**  
  Implemented and evaluated both global (whole image) and local (patch-based) models for wound segmentation. Combined their outputs using a convolutional fusion layer, as proposed in WSNet.

- **High-Resolution Image Training:**  
  Compared model performance across 256x256, 512x512, and 1024x1024 image resolutions. Achieved the best accuracy and boundary detail at 1024x1024.

- **Multiple Model Architectures:**  
  Tested U-Net, FPN, LinkNet, and PSPNet. FPN backbone gave the most stable and highest scores, used in the final combined model.

- **Loss Function Experimentation:**  
  Compared Dice, Focal, and Recall-based Cross-Entropy losses to address class imbalance and boundary detection. Focal loss provided the best results.

- **Comprehensive Data Augmentation:**  
  Employed horizontal flip, rotation, grid/optical distortion, motion blur, brightness, and contrast adjustments to simulate real-world clinical variability.

- **Robust Evaluation Protocol:**  
  Used segmentation metrics: Precision, Recall, Intersection over Union (IoU), and Dice Score. Achieved best Dice Score: **0.805** (combined FPN model).

- **Stable Training and Reproducibility:**  
  Maintained consistent batch sizes for train/test to ensure correct normalization and reliable results. Noted the importance of evaluation batch sizes for stable predictions.

- **Recommendations for Future Work:**  
  - Investigate learning rate schedulers and more training epochs for improved fusion and generalization.
  - Explore batch normalization for better batch-invariant segmentation.
  - Apply framework to larger and additional clinical datasets for further evaluation.

- **Clinical Impact:**  
  Demonstrated that fusing global and local segmentation predictions substantially improves wound detection—potentially assisting clinical wound assessment and treatment protocols.



## How to Run

1. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

2. **Prepare the data:**  
    > *Due to data privacy and clinical restrictions, the dataset used in this project cannot be shared publicly.  
    Please follow the formatting and directory structure as outlined in the report if using your own or test data.*

3. **Train the models:**
    ```
    python src/FPN_global_model_training_code.py
    python src/FPN_local_model_training_code.py
    python src/FPN_combined_model_training_code.py
    ```

4. **Review outputs:**  
    Training logs and model checkpoints are saved in the `logs/` directory upon completion.


## Customizing Model Architecture

To train with other segmentation architectures, simply edit the model definition in each training script. For example:

- In `FPN_global_model_training_code.py` and `FPN_local_model_training_code.py`:
    ```
    model = smp.FPN(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    ```
    Replace `smp.FPN` with other supported models, such as `smp.Unet`, `smp.Linknet`, or `smp.PSPNet`.

- In `FPN_combined_model_training_code.py`, within `CombinedModel`:
    ```
    class CombinedModel(nn.Module):
        def __init__(self):
            super(CombinedModel, self).__init__()
            self.global_model = smp.FPN(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
            self.local_model  = smp.FPN(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    ```
    Replace `smp.FPN` with any backbone supported by segmentation-models-pytorch.

*See the [segmentation-models-pytorch docs](https://github.com/qubvel/segmentation_models.pytorch) for more options and details.*

## What I Learned / Skills Gained

- Built and trained deep learning segmentation models with PyTorch and segmentation-models-pytorch.
- Applied and interpreted key evaluation metrics: IoU, recall, precision, Dice score.
- Implemented and compared advanced loss functions: Dice, focal, and recall cross-entropy losses.
- Managed large-scale model training on an HPC cluster equipped with CUDA GPUs.
- Applied extensive data augmentation for robust training.
- Explored state-of-the-art segmentation architectures: U-Net, FPN, LinkNet, PSPNet.
- Designed global, local, and combined (fusion) training approaches for improved accuracy.
- Tuned key hyperparameters (learning rates, batch sizes, epochs, optimizers) for optimal results.
- Systematically tested different image resolutions and training methods for thorough experimentation.
- Documented workflows, results, and produced technical reports for reproducibility.

## Reference

Oota et al., "WSNet: Towards An Effective Method for Wound Image Segmentation," WACV 2023.

---

*For full experimental setup, implementation details, and results, please refer to [Project_Report.pdf](Project_Report.pdf).*  

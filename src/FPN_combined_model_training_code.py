import os
import cv2
import albumentations as A
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2  # Import ToTensorV2 for conversion

# Load the trained model
local_model_save_path = '/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/LOCAL_FPN_FINAL_epoch_50.pth'
global_model_save_path = '/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/GLOBAL_FPN_FINAL_epoch_50.pth'

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your evaluation dataset and dataloader
# Assuming you already have the test dataset and dataloader defined (test_dataset and test_loader)
# If not, you can define them similar to how you did during training
test_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/test/images/'
test_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/test/masks/'
train_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/train/images/'
train_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/train/masks/'
val_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/validation/images/'
val_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/validation/masks/'

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),  # Random rotation
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),  # Optical distortion
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),  # Grid distortion
        A.MotionBlur(blur_limit=5, p=0.5),  # Blur
        A.RandomBrightnessContrast(p=0.5),  # Random brightness/contrast
        A.Transpose(p=0.5),  # Transpose (switch height/width)
        ToTensorV2()  # Convert image and mask to tensors
    ], additional_targets={'mask': 'mask'})  # Ensure augmentations are applied to both image and mask

# Test transformations (no augmentations, just tensor conversion)
def get_val_test_transform():
    return A.Compose([
        ToTensorV2()  # Convert image and mask to PyTorch tensors
    ], additional_targets={'mask': 'mask'})

def split_into_patches(image, patch_size=256):
    # Ensure image is a tensor
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)

    # Get the shape of the image
    _, h, w = image.shape  # shape should be (C, H, W)
    
    # Create a list to store patches
    patches = []

    # Iterate over the height and width to extract patches
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Extract the patch
            patch = image[:, i:i + patch_size, j:j + patch_size]  # Keep channel dimension
            if patch.shape[1] == patch_size and patch.shape[2] == patch_size:  # Ensure patch size
                patches.append(patch)

    # Convert the list of patches to a tensor and return it
    return torch.stack(patches)  # Stack patches into a tensor

def reassemble_patches(patches, batch_size):
    """ Reassemble the patches back into the original image """
    _, num_patches, channels, patch_height, patch_width = patches.shape
    
    # Calculate the number of patches per side assuming a square layout
    rows = cols = int(num_patches ** 0.5)  # This should be the number of patches per image (if square)
    
    # Adjust the reassembled image size based on the number of rows and cols
    reassembled_images = torch.zeros((batch_size, channels, rows * patch_height, cols * patch_width), device=patches.device)
    
    # Reassemble the patches
    for b in range(batch_size):
        for i in range(rows):
            for j in range(cols):
                patch_idx = i * cols + j
                if patch_idx < num_patches:  # Ensure we don't access out of bounds
                    reassembled_images[b, :, i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = patches[b, patch_idx]

    return reassembled_images


class WoundDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = os.listdir(image_dir)
        self.mask_files = os.listdir(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Read image and mask
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # Use grayscale for masks

        # Ensure the mask is binary (0 or 1)
        mask = np.where(mask > 128, 1, 0).astype(np.float32)  # Keep mask as a NumPy array

        original_image_shape = image.shape  # Store original shape before transformation
        original_mask_shape = mask.shape    # Store original shape of the mask
        
        # Apply transformations (if any)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # PyTorch tensor after transformation
            mask = augmented['mask']    # PyTorch tensor after transformation
            
        # Store the original mask before patching
        original_mask = torch.tensor(mask).clone()  # Convert mask to tensor and then clone it
        original_image = torch.tensor(image).clone()
        
        # Ensure the mask has a channel dimension (C, H, W)
        if mask.ndim == 2:
            mask = torch.unsqueeze(mask, 0)  # Convert shape from (H, W) to (1, H, W)
            
        # Split image and mask into 16 patches of 256x256
        image_patches = split_into_patches(image, patch_size=256)  # Already tensors
        mask_patches = split_into_patches(mask, patch_size=256)  # Already tensors

        num_patches = image_patches.size(0)  # Get the number of patches created

        return image_patches, mask_patches, original_mask_shape, original_mask, original_image, num_patches

train_dataset = WoundDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=get_train_transform())
val_dataset = WoundDataset(image_dir=val_image_dir, mask_dir=val_mask_dir, transform=get_val_test_transform())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)

# Initialize the test dataset and dataloader
test_dataset = WoundDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, transform=get_val_test_transform())
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        
        self.global_model = smp.FPN(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
        self.local_model = smp.FPN(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
        
        self.global_model.load_state_dict(torch.load(global_model_save_path, map_location=device))
        self.local_model.load_state_dict(torch.load(local_model_save_path, map_location=device))

        self.conv_layer = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1) 



    def forward(self, image_patches, original_image):
        # Reshape image_patches to (batch_size * num_patches, channels, height, width)
        
        batch_size, num_patches, channels, height, width = image_patches.shape
        reshaped_patches = image_patches.view(-1, channels, height, width)  # Reshape to (batch_size * num_patches, channels, height, width)
        
        # Convert reshaped_patches to float32 to match the model's expected input type
        reshaped_patches = reshaped_patches.float()

        # Forward pass for the local model
        local_output = self.local_model(reshaped_patches)
        local_output = local_output.view(batch_size, num_patches, -1, height, width)
        reassembled_preds = reassemble_patches(local_output,batch_size)

        global_output = self.global_model(original_image)

        # Concatenate reassembled_preds and global_output along the channel dimension (dim=1)
        combined_output = torch.cat((reassembled_preds, global_output), dim=1)

        # Pass the concatenated output through the conv_layer (in_channels=2, out_channels=1)
        final_output = self.conv_layer(combined_output)  # Shape: (batch_size, 1, H, W)

        return final_output


# Create an instance of the combined model
combined_model = CombinedModel()
combined_model.to(device)

loss_fn = smp.losses.FocalLoss(mode='binary')
optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.0001)


def evaluate_model_performance(loader, combined_model, device):
    combined_model.eval()
    metrics = {'precision': [], 'recall': [], 'iou': [], 'dice': []}

    with torch.no_grad():
        for image_patches, masks, original_mask_shape, original_mask, original_image, num_patches  in loader:  # Adjusted to unpack four values
            
            batch_size = image_patches.size(0)
            image_patches = image_patches.to(device)  # Move image patches to the device
            original_image = original_image.to(device)  # Move original image to the device            
            original_mask = original_mask.to(device)  # Ensure original_mask is on the same device
           
            outputs = combined_model(image_patches,original_image)

            preds = torch.sigmoid(outputs).squeeze(1) > 0.5
            
            preds = preds.long()  # Convert preds to long without adding a channel dimension

            # Calculate true positives, false positives, false negatives, true negatives
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), original_mask.long(), mode='binary')

            # Calculate each metric and store
            metrics['precision'].append(smp.metrics.precision(tp, fp, fn, tn, reduction='micro').item())
            metrics['recall'].append(smp.metrics.recall(tp, fp, fn, tn, reduction='micro').item())
            metrics['iou'].append(smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item())
            metrics['dice'].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro').item())

    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics

# Training and validation loop
num_epochs = 75

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    combined_model.train()  # Set model to training mode
    running_loss = 0.0
    for image_patches, masks, original_mask_shape, original_mask, original_image, num_patches in train_loader:
        
        image_patches = image_patches.to(device)  # Move image patches to the device
        masks = masks.to(device, dtype=torch.float32)
        original_image = original_image.to(device)  # Move original image to the device            
        original_mask = original_mask.to(device)  # Ensure original_mask is on the same device

        outputs = combined_model(image_patches,original_image)

        loss = loss_fn(outputs, original_mask)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Training Loss: {avg_train_loss}")

    test_metrics = evaluate_model_performance(test_loader, combined_model, device)
    print(f"Test Metrics at after Epoch {epoch + 1}: Precision: {test_metrics['precision']:.4f}, "f"Recall: {test_metrics['recall']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice:{test_metrics['dice']:.4f}")
    
test_metrics = evaluate_model_performance(test_loader, combined_model, device)
print(f"Test Metrics: Precision: {test_metrics['precision']:.4f}, "f"Recall: {test_metrics['recall']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}")

print("Training and evaluation complete!")
main_model_save_path = f'/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/COMBINED_singleconvolayer_50.pth'
torch.save(combined_model.state_dict(), main_model_save_path)
print(f"Model saved after epoch {epoch+1} to {main_model_save_path}")

import os
import cv2
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2  # Import ToTensorV2 for conversion

# Directories for images and masks
train_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/train/images/'
train_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/train/masks/'
val_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/validation/images/'
val_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/validation/masks/'
test_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/test/images/'
test_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/test/masks/'

# Transformations for train dataset with ToTensorV2
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

# Transformation for validation and test sets (no augmentation, just ToTensorV2)
def get_val_test_transform():
    return A.Compose([
        ToTensorV2()  # Only convert image and mask to tensors without any augmentation
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

# Dataset class
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
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Read image and mask
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Use grayscale for masks

        # Ensure the mask is binary (0 or 1)
        mask = np.where(mask > 128, 1, 0).astype(np.float32)  # Keep mask as a NumPy array

        original_image_shape = image.shape  # Store original shape before transformation
        original_mask_shape = mask.shape    # Store original shape of the mask
        
        
        ####################
        #print(f"Original image shape: {original_image_shape}")  # Print original image shape
        #print(f"Original mask shape: {original_mask_shape}")    # Print original mask shape

        # Apply transformations (if any)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # This will be a PyTorch tensor
            mask = augmented['mask']    # This will be a PyTorch tensor
        
        # Store the original mask before patching
        original_mask = mask.clone()  # Make a clone of the mask tensor to store the unpatched mask
        
        # Ensure the mask has a channel dimension (C, H, W)
        if mask.ndim == 2:
            mask = torch.unsqueeze(mask, 0)  # Convert shape from (H, W) to (1, H, W)

        # Split image and mask into 16 patches of 256x256
        image_patches = split_into_patches(image, patch_size=256)  # Already tensors
        mask_patches = split_into_patches(mask, patch_size=256)  # Already tensors
        
        ######################
        #print(f"Patches image shape: {image_patches.shape}")  # Print patch shape
        #print(f"Patches mask shape: {mask_patches.shape}")    # Print patch shape

        return image_patches, mask_patches,original_mask_shape,original_mask   # Return original shapes


# Initialize datasets and dataloaders
train_dataset = WoundDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=get_train_transform())
val_dataset = WoundDataset(image_dir=val_image_dir, mask_dir=val_mask_dir, transform=get_val_test_transform())
test_dataset = WoundDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, transform=get_val_test_transform())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Example of stitching back a sample image and mask from the dataset
image_patches, mask_patches,original_mask_shape,original_mask   = train_dataset[0]


# FPN model with ResNet34 backbone
model = smp.FPN(encoder_name="resnet34", 
                encoder_weights="imagenet", 
                in_channels=3,  
                classes=1,  
                activation=None)  

#model = nn.DataParallel(model)
#model.to(device)

# Loss function and optimizer
loss_fn = smp.losses.FocalLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def reassemble_patches(patches):
    """
    Reassembles patches into full images for a batch of images.
    
    Args:
        patches (Tensor): Tensor of shape (N, channels, patch_size, patch_size)
                          containing the patches in order for each image in the batch.
        
    Returns:
        Tensor: Reassembled images of shape (batch_size, channels, 4*patch_size, 4*patch_size).
    """
    # Get the number of patches, channels, and patch size
    total_patches, channels, patch_size, _ = patches.shape
    
    # Calculate the batch size based on the total number of patches
    batch_size = total_patches // 16  # Integer division to get the number of images
    
    # Initialize a tensor to hold the reassembled images
    reassembled_images = torch.zeros((batch_size, channels, 4 * patch_size, 4 * patch_size), device=patches.device)
    
    # Loop over each image in the batch
    for b in range(batch_size):
        for i in range(4):  # Rows of patches
            for j in range(4):  # Columns of patches
                patch_idx = b * 16 + i * 4 + j  # Calculate the index for the current patch
                if patch_idx < total_patches:  # Check to avoid out of bounds access
                    reassembled_images[b, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches[patch_idx]
    
    return reassembled_images


def evaluate_model_performance(loader, model, device):
    model.eval()
    metrics = {'precision': [], 'recall': [], 'iou': [], 'dice': []}

    with torch.no_grad():
        for images, masks,original_mask_shape,original_mask  in loader:  # Adjusted to unpack four values
            
            images = images.view(-1, 3, 256, 256)  # Flatten the first two dimensions (batch_size * num_patches, C, H, W)
            masks = masks.view(-1, 1, 256, 256)    # Flatten the first two dimensions (batch_size * num_patches, 1, H, W)
            
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            
            #print(f"Model output shape: {outputs.shape}")
            
            # Example: Patches shape is (128, 3, 256, 256) for a batch of 8 images
            # Reassemble the predicted patches into full-sized images
            reassembled_preds = reassemble_patches(outputs)
            
            # Print the shape of the reassembled images
            #print("The reassembled prediction size is: ", reassembled_preds.size())
            
            preds = torch.sigmoid(reassembled_preds).squeeze(1) > 0.5
            
            preds = preds.long()  # Convert preds to long without adding a channel dimension
            
            original_mask = original_mask.to(device)  # Ensure original_mask is on the same device


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


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    eval_loss = 0.0
    with torch.no_grad():  # No gradient calculation
        for images, masks,original_mask_shape,original_mask   in data_loader:
            # Reshape image and mask tensors
            images = images.view(-1, 3, 256, 256)  # Flatten the first two dimensions (batch_size * num_patches, C, H, W)
            masks = masks.view(-1, 1, 256, 256)    # Flatten the first two dimensions (batch_size * num_patches, 1, H, W)

            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(images)
            #print(f"Model output shape during validation: {outputs.shape}")
            loss = loss_fn(outputs, masks)
            eval_loss += loss.item()
    
    return eval_loss / len(data_loader)


# Training and validation loop
num_epochs = 50


for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    model.train()  # Set model to training mode
    running_loss = 0.0
    for batch_idx, (images, masks,original_mask_shape,original_mask) in enumerate(train_loader):
        images = images.view(-1, 3, 256, 256)  # Flatten the first two dimensions (batch_size * num_patches, C, H, W)
        masks = masks.view(-1, 1, 256, 256)    # Flatten the first two dimensions (batch_size * num_patches, 1, H, W)
        
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        
        # Forward pass
        outputs = model(images)
        #print(f"Model output shape during training: {outputs.shape}")
        loss = loss_fn(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Training Loss: {avg_train_loss}")
    
    # Validate the model after each epoch
    val_loss = evaluate_model(model, val_loader, loss_fn, device)
    print(f"Validation Loss: {val_loss}")

    
    if (epoch + 1) % 5 == 0:  # Check if the epoch is a multiple of 10
        model_save_path = f'/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/LOCAL_FPN_FINAL_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved after epoch {epoch+1} to {model_save_path}")
        test_metrics = evaluate_model_performance(test_loader, model, device)
        print(f"Test Metrics at Epoch {epoch + 1}: Precision: {test_metrics['precision']:.4f}, "f"Recall: {test_metrics['recall']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice:{test_metrics['dice']:.4f}")

    
# Evaluate on test set after training
print("\nEvaluating on test set...")
# Evaluate metrics on test set
# Call evaluate_model with all required arguments (including loss_fn)
test_loss = evaluate_model(model, test_loader, loss_fn, device)
print(f"Test Loss: {test_loss:.4f}")


test_metrics = evaluate_model_performance(test_loader, model, device)
print(f"Test Metrics at Epoch {epoch + 1}: Precision: {test_metrics['precision']:.4f}, "f"Recall: {test_metrics['recall']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}")


print("Training and evaluation complete!")
main_model_save_path = f'/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/LOCAL_FPN_FINAL_epoch_50.pth'
torch.save(model.state_dict(), main_model_save_path)
print(f"Model saved after epoch {epoch+1} to {main_model_save_path}")


# Stitch the patches back together

#stitched_mask = stitch_patches(mask_patches, original_mask_shape, patch_size=256)

#print(f"Stitched mask shape: {stitched_mask.shape}")    # Print stitched mask shape





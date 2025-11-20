import os
import cv2
import albumentations as A
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import torch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

# Directories for images and masks
train_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/train/images/'
train_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/train/masks/'
val_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/validation/images/'
val_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/validation/masks/'
test_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/test/images/'
test_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/test/masks/'

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),  # Random rotation
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),  # Optical distortion
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),  # Grid distortion
        A.MotionBlur(blur_limit=5, p=0.5),  # Blur
        A.RandomBrightnessContrast(p=0.5),  # Random brightness/contrast
        A.Transpose(p=0.5),  # Transpose (switch height/width)
    ], additional_targets={'mask': 'mask'})  # To apply the same augmentation to the mask

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

        # Read image and mask (image is uint8, mask is grayscale)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # uint8 by default
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Use grayscale for masks (uint8)

        # Ensure the mask is binary (0 or 1)
        mask = np.where(mask > 128, 1, 0).astype(np.float32)
        image = image.astype(np.float32)   # Normalize image
        
        # Apply transformations (if any)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1))  # From (H, W, C) to (C, H, W)

        # Return as uint8 (image) and float32 (mask)
        return image, mask.astype(np.float32)

# Initialize datasets and dataloaders
train_dataset = WoundDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=get_train_transform())
val_dataset = WoundDataset(image_dir=val_image_dir, mask_dir=val_mask_dir)
test_dataset = WoundDataset(image_dir=test_image_dir, mask_dir=test_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Print shape of random 3 images and masks
def print_random_image_mask_info(dataset):
    print("Random 3 images and masks:")

    random_indices = random.sample(range(len(dataset)), 3)  # Pick 3 random indices

    for idx in random_indices:
        image, mask = dataset[idx]

        print(f"Image {idx} shape: {image.shape}")
        print(f"Mask {idx} shape: {mask.shape}")

        # Count the pixel values in the mask using Counter
        mask_counter = Counter(mask.flatten())
        print(f"Mask {idx} pixel value counts: {mask_counter}")
        print("-" * 50)

# Test the function with train dataset
#print_random_image_mask_info(train_dataset)

# FPN model with ResNet34 backbone
model = smp.FPN(encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,  # RGB image (3 channels)
                classes=1,  # Binary mask
                activation=None)

# Loss function and optimizer
loss_fn = smp.losses.FocalLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate_model_performance(loader, model, device):
    model.eval()
    metrics = {'precision': [], 'recall': [], 'iou': [], 'dice': []}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            preds = torch.sigmoid(outputs).squeeze(1) > 0.5

            # Calculate true positives, false positives, false negatives, true negatives
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='binary')

            # Calculate each metric and store
            metrics['precision'].append(smp.metrics.precision(tp, fp, fn, tn, reduction='micro').item())
            metrics['recall'].append(smp.metrics.recall(tp, fp, fn, tn, reduction='micro').item())
            metrics['iou'].append(smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item())
            metrics['dice'].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro').item())

    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics

# Function to evaluate model
def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    eval_loss = 0.0
    with torch.no_grad():  # No gradient calculation
        for images, masks in data_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            eval_loss += loss.item()

    return eval_loss / len(data_loader)

# Training and validation loop
num_epochs = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    model.train()  # Set model to training mode
    running_loss = 0.0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        # Forward pass
        outputs = model(images)
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

    if (epoch + 1) % 10 == 0:  # Check if the epoch is a multiple of 10
        model_save_path = f'/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/GLOBAL_FPN_FINAL_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved after epoch {epoch+1} to {model_save_path}")
        test_metrics = evaluate_model_performance(test_loader, model, device)
        print(f"Test Metrics at Epoch {epoch + 1}: Precision: {test_metrics['precision']:.4f}, "
              f"Recall: {test_metrics['recall']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice:{test_metrics['dice']:.4f}")


# Evaluate on test set after training
print("\nEvaluating on test set...")
# Evaluate metrics on test set
test_loss = evaluate_model(model, test_loader, loss_fn, device)
print(f"Test Loss: {test_loss:.4f}")

test_metrics = evaluate_model_performance(test_loader, model, device)
print(f"Test Metrics at Epoch {epoch + 1}: Precision: {test_metrics['precision']:.4f}, "
      f"Recall: {test_metrics['recall']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}")

print("Training and evaluation complete!")

main_model_save_path = f'/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/GLOBAL_FPN_FINAL_epoch_50.pth'
torch.save(model.state_dict(), main_model_save_path)
print(f"Model saved after epoch {epoch+1} to {main_model_save_path}")







# import os
# import cv2
# import albumentations as A
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from collections import Counter
# import random
# import torch
# import segmentation_models_pytorch as smp
# from albumentations.pytorch import ToTensorV2

# # Directories for images and masks
# train_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/train/images/'
# train_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/train/masks/'
# val_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/validation/images/'
# val_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/validation/masks/'
# test_image_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/test/images/'
# test_mask_dir = '/home/hpc/iwso/iwso151h/DATASET_NOAUG/test/masks/'

# def get_train_transform():
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.Rotate(limit=45, p=0.5),  # Random rotation
#         A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),  # Optical distortion
#         A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),  # Grid distortion
#         A.MotionBlur(blur_limit=5, p=0.5),  # Blur
#         A.RandomBrightnessContrast(p=0.5),  # Random brightness/contrast
#         A.Transpose(p=0.5),  # Transpose (switch height/width)
#     ], additional_targets={'mask': 'mask'})  # To apply the same augmentation to the mask

# # Dataset class
# class WoundDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.image_files = os.listdir(image_dir)
#         self.mask_files = os.listdir(mask_dir)
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         # Load image and mask
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

#         # Read image and mask
#         image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Use grayscale for masks

#         # Ensure the mask is binary (0 or 1)
#         mask = np.where(mask > 128, 1, 0).astype(np.float32)

#         # Apply transformations (if any)
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
            
#         image = np.transpose(image, (2, 0, 1))  # From (H, W, C) to (C, H, W)

#         return image, mask

# # Initialize datasets and dataloaders
# train_dataset = WoundDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=get_train_transform())
# val_dataset = WoundDataset(image_dir=val_image_dir, mask_dir=val_mask_dir)
# test_dataset = WoundDataset(image_dir=test_image_dir, mask_dir=test_mask_dir)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # Print shape of random 3 images and masks
# def print_random_image_mask_info(dataset):
#     print("Random 3 images and masks:")
    
#     random_indices = random.sample(range(len(dataset)), 3)  # Pick 3 random indices

#     for idx in random_indices:
#         image, mask = dataset[idx]
        
#         print(f"Image {idx} shape: {image.shape}")
#         print(f"Mask {idx} shape: {mask.shape}")
        
#         # Count the pixel values in the mask using Counter
#         mask_counter = Counter(mask.flatten())
#         print(f"Mask {idx} pixel value counts: {mask_counter}")
#         print("-" * 50)

# # Test the function with train dataset
# #print_random_image_mask_info(train_dataset)

# # FPN model with ResNet34 backbone
# model = smp.FPN(encoder_name="resnet34", 
#                 encoder_weights="imagenet", 
#                 in_channels=3,  
#                 classes=1,  
#                 activation=None)  

# # Loss function and optimizer
# loss_fn = smp.losses.FocalLoss(mode='binary')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# def evaluate_model_performance(loader, model, device):
#     model.eval()
#     metrics = {'precision': [], 'recall': [], 'iou': [], 'dice': []}

#     with torch.no_grad():
#         for images, masks in loader:
#             images = images.to(device, dtype=torch.float32)
#             masks = masks.to(device, dtype=torch.float32)

#             outputs = model(images)
#             preds = torch.sigmoid(outputs).squeeze(1) > 0.5

#             # Calculate true positives, false positives, false negatives, true negatives
#             tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='binary')

#             # Calculate each metric and store
#             metrics['precision'].append(smp.metrics.precision(tp, fp, fn, tn, reduction='micro').item())
#             metrics['recall'].append(smp.metrics.recall(tp, fp, fn, tn, reduction='micro').item())
#             metrics['iou'].append(smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item())
#             metrics['dice'].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro').item())

#     # Average metrics
#     avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
#     return avg_metrics

# # Function to evaluate model
# def evaluate_model(model, data_loader, loss_fn, device):
#     model.eval()  # Set model to evaluation mode
#     eval_loss = 0.0
#     with torch.no_grad():  # No gradient calculation
#         for images, masks in data_loader:
#             images = images.to(device, dtype=torch.float32)
#             masks = masks.to(device, dtype=torch.float32)
            
#             # Forward pass
#             outputs = model(images)
#             loss = loss_fn(outputs, masks)
#             eval_loss += loss.item()
    
#     return eval_loss / len(data_loader)

# # Training and validation loop
# num_epochs = 50

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")
    
#     model.train()  # Set model to training mode
#     running_loss = 0.0
#     for batch_idx, (images, masks) in enumerate(train_loader):
#         images = images.to(device, dtype=torch.float32)
#         masks = masks.to(device, dtype=torch.float32)
        
#         # Forward pass
#         outputs = model(images)
#         loss = loss_fn(outputs, masks)
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
    
#     avg_train_loss = running_loss / len(train_loader)
#     print(f"Training Loss: {avg_train_loss}")
    
#     # Validate the model after each epoch
#     val_loss = evaluate_model(model, val_loader, loss_fn, device)
#     print(f"Validation Loss: {val_loss}")
    
#     if (epoch + 1) % 10 == 0:  # Check if the epoch is a multiple of 10
#         model_save_path = f'/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/GLOBAL_FPN_FINAL_epoch_{epoch+1}.pth'
#         torch.save(model.state_dict(), model_save_path)
#         print(f"Model saved after epoch {epoch+1} to {model_save_path}")
#         test_metrics = evaluate_model_performance(test_loader, model, device)
#         print(f"Test Metrics at Epoch {epoch + 1}: Precision: {test_metrics['precision']:.4f}, "f"Recall: {test_metrics['recall']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice:{test_metrics['dice']:.4f}")
        
        
        
        

# # Evaluate on test set after training
# print("\nEvaluating on test set...")
# # Evaluate metrics on test set
# # Call evaluate_model with all required arguments (including loss_fn)
# test_loss = evaluate_model(model, test_loader, loss_fn, device)
# print(f"Test Loss: {test_loss:.4f}")

# test_metrics = evaluate_model_performance(test_loader, model, device)
# print(f"Test Metrics at Epoch {epoch + 1}: Precision: {test_metrics['precision']:.4f}, "f"Recall: {test_metrics['recall']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}")

# print("Training and evaluation complete!")

# main_model_save_path = f'/home/hpc/iwso/iwso151h/FPN_GLOBAL_LOCAL/GLOBAL_FPN_FINAL_epoch_50.pth'
# torch.save(model.state_dict(), main_model_save_path)
# print(f"Model saved after epoch {epoch+1} to {main_model_save_path}")

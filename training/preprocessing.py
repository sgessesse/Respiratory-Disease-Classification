# Data Preprocessing for Chest X-Ray Classification
#
# This notebook handles the loading and preprocessing of the chest X-ray dataset. The dataset is divided into training, validation, and test sets. Each set contains images and their corresponding labels. The images are normalized and converted to PyTorch tensors for model training.
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset # Added Dataset
from pathlib import Path
import cv2 # Added for CLAHE
import time # Added for timing calculation

# Define the custom Dataset
class XRayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (numpy.ndarray): Array of images (N, H, W, C).
            labels (numpy.ndarray): Array of labels (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply CLAHE here if needed, before other transforms
        # Assuming image is uint8 (0-255) and single channel for CLAHE
        # If image is RGB, apply CLAHE to the L channel in LAB color space or grayscale
        # For simplicity, let's assume grayscale or apply to each channel if RGB
        if self.transform:
             # If image is RGB (H, W, 3), apply CLAHE channel-wise or convert to grayscale first
             # Example for grayscale (assuming input is H, W, 1 or H, W)
             # if image.ndim == 2 or image.shape[2] == 1:
             #    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
             #    image = clahe.apply(image.astype(np.uint8))
             #    image = np.expand_dims(image, axis=-1) # Add channel dim back if needed
             # Example for RGB (applying to L channel of LAB)
             if image.shape[2] == 3:
                 lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
                 l, a, b = cv2.split(lab)
                 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                 cl = clahe.apply(l)
                 limg = cv2.merge((cl,a,b))
                 image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
             # Convert back to PIL Image for torchvision transforms if needed, or handle tensor conversion directly
             # image = Image.fromarray(image) # If transforms expect PIL

             # Apply the rest of the transforms
             image = self.transform(image) # Assumes transform handles tensor conversion

        return image, torch.tensor(label, dtype=torch.long)


def calculate_mean_std(image_array):
    """ Calculates mean and std deviation for image normalization. Assumes NCHW format after permute. """
    # Convert to float32 tensor, permute, and scale
    tensor = torch.tensor(image_array, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    mean = torch.mean(tensor, dim=[0, 2, 3])
    std = torch.std(tensor, dim=[0, 2, 3])
    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std: {std}")
    return mean.tolist(), std.tolist()


def load_data(path):
    """
    Load the dataset from the given path, calculate stats, and create Datasets.

    Args:
        path (Path): Path to the directory containing the dataset .npz files.

    Returns:
        train_ds (TensorDataset): Training dataset.
        val_ds (TensorDataset): Validation dataset.
        test_ds (TensorDataset): Test dataset.
    """
    try:
        train_data = np.load(path / "Dataset5_raw_train.npz")
        val_data = np.load(path / "Dataset5_raw_val.npz")
        test_data = np.load(path / "Dataset5_raw_test.npz")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Ensure .npz files are in {path}")
        raise # Re-raise the exception to stop execution if files are missing

    print("Loading data arrays...")
    train_images = train_data["image"] # Shape (N, H, W, C), assuming uint8
    val_images = val_data["image"]
    test_images = test_data["image"]
    # Ensure labels are 1D numpy arrays
    train_labels = train_data["image_label"].astype(np.int64).squeeze()
    val_labels = val_data["image_label"].astype(np.int64).squeeze()
    test_labels = test_data["image_label"].astype(np.int64).squeeze()
    print("Data arrays loaded.")

    # Calculate mean and std from training images (only need to do this once)
    print("Calculating normalization stats from training data...")
    start_time = time.time()
    # Ensure images are in range 0-1 for calculation if needed, or handle uint8 directly
    # For simplicity, calculate on the uint8 array and scale mean/std later if needed,
    # but it's better to calculate on float32/scaled data.
    # Let's calculate after scaling to 0-1 range.
    mean, std = calculate_mean_std(train_images)
    print(f"Stats calculation took {time.time() - start_time:.2f} seconds.")


    # Define transforms
    # Note: Input to transforms should ideally be PIL Image or Tensor.
    # If input is numpy array, ToTensor() handles it.
    # CLAHE needs to be applied before ToTensor if using cv2 on numpy array.
    # We will apply CLAHE within the Dataset __getitem__ method.

    train_transform = transforms.Compose([
        transforms.ToTensor(), # Converts numpy array (H, W, C) in range [0, 255] to Tensor (C, H, W) in range [0.0, 1.0]
        transforms.Resize((224, 224), antialias=True), # Resize after converting to tensor
        # Apply intensity augmentations on tensors
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2) # Adjust brightness and contrast
        ], p=0.5),
        # transforms.RandomApply([ # Optional: Add Gaussian Noise
        #     lambda x: x + torch.randn_like(x) * 0.01
        # ], p=0.2),
        transforms.Normalize(mean=mean, std=std) # Normalize using calculated stats
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Create custom Dataset instances
    print("Creating Dataset instances...")
    train_ds = XRayDataset(train_images, train_labels, transform=train_transform)
    val_ds = XRayDataset(val_images, val_labels, transform=val_test_transform)
    test_ds = XRayDataset(test_images, test_labels, transform=val_test_transform)
    print("Dataset instances created.")

    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    return train_ds, val_ds, test_ds

def create_data_loaders(train_ds, val_ds, test_ds, batch_size=64, num_workers=4): # Increased default num_workers
    """
    Create DataLoader objects for training, validation, and test datasets.

    Args:
        train_ds (Dataset): Training dataset (custom XRayDataset).
        val_ds (Dataset): Validation dataset (custom XRayDataset).
        test_ds (Dataset): Test dataset (custom XRayDataset).
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of subprocesses to use for data loading. Recommended: 2-4 per GPU or based on CPU cores.

    Returns:
        train_dl (DataLoader): DataLoader for training dataset.
        val_dl (DataLoader): DataLoader for validation dataset.
        test_dl (DataLoader): DataLoader for test dataset.
    """
    pin_memory = torch.cuda.is_available() # Pin memory if CUDA is available
    print(f"Using num_workers={num_workers}, pin_memory={pin_memory}")

    # Use persistent_workers=True if num_workers > 0 to avoid recreating workers every epoch (PyTorch >= 1.7)
    persistent_workers = num_workers > 0

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    # Keep validation/test batch size potentially larger, but consider GPU memory
    val_batch_size = batch_size * 2
    test_batch_size = batch_size * 2
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    print(f"Created DataLoaders with batch sizes: Train={batch_size}, Val={val_batch_size}, Test={test_batch_size}")
    return train_dl, val_dl, test_dl

# Example Usage (uncommented to allow direct execution for testing)
if __name__ == '__main__':
    # Correct path when running script directly from project root
    data_path = Path("raw_data")
    try:
        print("--- Running Preprocessing Test ---")
        # The load_data function itself expects the path relative to where it's called from.
        # When called from main.py (in notebooks/), "../raw_data" is correct.
        # When called from here (running preprocessing.py directly from root), "raw_data" is correct.
        # Let's adjust the call here for direct execution testing.
        # We pass the correct path relative to this script's execution context (project root).
        train_dataset, val_dataset, test_dataset = load_data(data_path)
        # Use a smaller batch size for testing to avoid memory issues if needed
        train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16, num_workers=0) # Use num_workers=0 for easier debugging if needed

        print("\n--- Testing DataLoader Iteration ---")
        # Example: Iterate over one batch of training data
        for i, (images, labels) in enumerate(train_loader):
            print(f"Batch {i+1} shapes: Images={images.shape}, Labels={labels.shape}")
            print(f"Batch {i+1} dtypes: Images={images.dtype}, Labels={labels.dtype}")
            # Check image value range (should be roughly normalized around 0)
            print(f"Batch {i+1} image stats: Min={images.min():.2f}, Max={images.max():.2f}, Mean={images.mean():.2f}")
            # Check label range if necessary
            # print("Label range:", labels.min().item(), labels.max().item())
            if i >= 0: # Only show first batch for brevity
                 break
        print("\n--- Preprocessing Test Completed ---")

    except Exception as e:
        print(f"\n--- An error occurred during example usage ---")
        import traceback
        traceback.print_exc()

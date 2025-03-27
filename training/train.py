# Model Training for Chest X-Ray Classification
#
#
#
# This notebook handles the training of the model. It uses the training and validation datasets to train the model and implements early stopping to prevent overfitting.
import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau # Old scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR # New scheduler
import torch.nn as nn
import numpy as np
import time # Import time for tracking duration

def train_model(model, train_dl, val_dl, epochs=50, patience=7, learning_rate=0.001, weight_decay=1e-4, label_smoothing=0.1, grad_clip_value=1.0):
    """
    Train the model using the given datasets with early stopping, cosine annealing LR, gradient clipping, and label smoothing.

    Args:
        model (nn.Module): Model to be trained.
        train_dl (DataLoader): DataLoader for the training dataset.
        val_dl (DataLoader): DataLoader for the validation dataset.
        epochs (int): Maximum number of epochs to train.
        patience (int): Number of epochs to wait for improvement before early stopping.
        learning_rate (float): Initial learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for AdamW optimizer.
        label_smoothing (float): Amount of label smoothing for CrossEntropyLoss.
        grad_clip_value (float): Max norm for gradient clipping.


    Returns:
        model (nn.Module): Trained model loaded with the best weights based on validation loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    # Use label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    # Use Cosine Annealing scheduler
    # T_max is typically the total number of iterations (steps) per epoch * number of epochs, or just number of epochs
    # Let's set T_max to total epochs for simplicity, it will complete one cycle over the training duration.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) # eta_min is the minimum learning rate
    print(f"Using CosineAnnealingLR scheduler with T_max={epochs}, eta_min={1e-6}")


    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Weighted by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_dl.dataset)
        train_acc = correct_train / total_train

        # Validation phase
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item() * inputs.size(0) # Weighted by batch size
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        current_val_loss = running_val_loss / len(val_dl.dataset)
        val_acc = correct_val / total_val
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"LR: {current_lr:.6f}.. " # Log LR
              f"Train Loss: {train_loss:.4f}.. Train Acc: {train_acc:.4f}.. "
              f"Val Loss: {current_val_loss:.4f}.. Val Acc: {val_acc:.4f}.. "
              f"Duration: {epoch_duration:.2f}s")


        # Check for improvement for early stopping
        if current_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {current_val_loss:.4f}). Saving model ...")
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            # Save the best model state
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")


        # Early stopping check
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

        # Step the Cosine Annealing scheduler after each epoch
        scheduler.step()

    total_time = time.time() - start_time
    print(f'\nTraining finished in {total_time // 60:.0f}m {total_time % 60:.0f}s')

    # Load the best model weights
    if best_model_state:
        print(f"Loading best model state with validation loss: {best_val_loss:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state saved. Returning the model from the last epoch.")

    return model

# Example Usage (optional, can be commented out if used as a module)
# if __name__ == '__main__':
#     # This requires preprocessing, model definition to be available
#     from preprocessing import load_data, create_data_loaders
#     from model import create_model
#     from pathlib import Path
#
#     data_path = Path("../raw_data")
#     batch_size = 32
#     num_epochs = 5 # Keep low for example
#
#     try:
#         train_dataset, val_dataset, test_dataset = load_data(data_path)
#         train_loader, val_loader, _ = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
#
#         # Create a model instance
#         model_instance = create_model(num_classes=5)
#
#         # Train the model
#         trained_model_instance = train_model(model_instance, train_loader, val_loader, epochs=num_epochs, patience=3)
#
#         print("Training example completed.")
#         # Optionally save the trained model
#         # torch.save(trained_model_instance.state_dict(), 'example_trained_model.pth')
#
#     except Exception as e:
#         print(f"An error occurred during training example: {e}")

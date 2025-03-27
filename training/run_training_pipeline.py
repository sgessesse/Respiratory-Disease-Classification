# Classifying Respiratory Diseases from Chest X-Ray Images Using Machine Learning
#
# Main script to run the full pipeline: preprocessing, training, evaluation, and model saving.

from preprocessing import load_data, create_data_loaders
from model import create_model
from train import train_model
from evaluate import evaluate_model
# from predict import Predictor # Assuming Predictor is defined elsewhere if needed
from pathlib import Path
import torch
import os

# Define paths and parameters
# Construct path relative to this script's location
script_dir = Path(__file__).parent
data_path = (script_dir / "../raw_data").resolve() # Use absolute path for data
model_save_dir = (script_dir / "../app/model").resolve() # Use absolute path for saving model
evaluation_output_dir = script_dir / "evaluation_results" # Save evaluation in notebooks/evaluation_results

print(f"Using data path: {data_path}")
print(f"Model save directory: {model_save_dir}")
print(f"Evaluation results directory: {evaluation_output_dir}")

# Hyperparameters
batch_size = 64
epochs = 50 # Adjust as needed
num_classes = 5

# --- Main execution block ---
if __name__ == '__main__':
    # Data Preprocessing
    print("--- Starting Data Preprocessing ---")
    train_ds, val_ds, test_ds = load_data(data_path)
    # Use num_workers=0 for stability on Windows
    print("--- Creating DataLoaders with num_workers=0 ---")
    train_dl, val_dl, test_dl = create_data_loaders(train_ds, val_ds, test_ds, batch_size, num_workers=0)
    print("--- Data Preprocessing Finished ---")

    # Model Architecture
    print("--- Creating Model ---")
    model = create_model(num_classes=num_classes)
    print("--- Model Created ---")

    # Training Process
    print("--- Starting Training ---")
    # Pass hyperparameters defined above
    trained_model = train_model(model, train_dl, val_dl, epochs=epochs) # Use defined epochs
    print("--- Training Finished ---")

    # Evaluation
    print("--- Starting Evaluation ---")
    # Pass the specific output directory
    evaluate_model(trained_model, test_dl, output_dir=evaluation_output_dir)
    print("--- Evaluation Finished ---")

    # Save the trained model
    print("--- Saving Model ---")
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = model_save_dir / "my_trained_model.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print("--- Model Saved ---")

    print("\n--- Pipeline Completed ---")

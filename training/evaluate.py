# Model Evaluation for Chest X-Ray Classification
#
# This notebook handles the evaluation of the trained model. It calculates the accuracy, F1 score,
# precision, recall, and generates a confusion matrix, saving results to files.
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd # Added for CSV saving
import os # Added for path handling
from pathlib import Path # Added for path handling

def evaluate_model(model, test_dl, output_dir="evaluation_results"):
    """
    Evaluate the model on the test dataset and save results.

    Args:
        model (nn.Module): Trained model.
        test_dl (DataLoader): DataLoader for the test dataset.
        output_dir (str): Directory to save evaluation results (CSV and PNG).
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation results to: {output_path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    model.to(device)

    model.eval() # Ensure model is in evaluation mode
    true_labels = []
    predicted_labels = []

    with torch.no_grad(): # Context manager for inference
        for inputs, labels in test_dl:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    # Use zero_division=0 to handle cases where a class might have no predicted samples
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    print(f"Accuracy on Test Set: {accuracy:.4f}")
    print(f"Precision (Weighted) on Test Set: {precision:.4f}")
    print(f"Recall (Weighted) on Test Set: {recall:.4f}")
    print(f"F1 Score (Weighted) on Test Set: {f1:.4f}")

    # Save metrics to CSV
    metrics_data = {
        'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)'],
        'Score': [accuracy, precision, recall, f1]
    }
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = output_path / "evaluation_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

    # Detailed classification report (optional print)
    classes = ["COVID-19", "Lung-Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"]
    print("\nClassification Report:")
    # Use zero_division=0 here as well
    print(classification_report(true_labels, predicted_labels, target_names=classes, zero_division=0))

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, predicted_labels)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap WITHOUT Seaborn's automatic annotations
    sns.heatmap(cm, cmap='Blues', ax=ax, annot=False, cbar=False)

    # Manually set axis ticks and labels
    ax.set_xticks(np.arange(len(classes)) + 0.5)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(classes)) + 0.5)
    ax.set_yticklabels(classes, rotation=0)

    # Add annotations manually to ensure they appear
    for i in range(len(classes)):
        for j in range(len(classes)):
            # Choose text color based on background
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha='center', va='center', color=text_color)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Save the plot
    png_path = output_path / "confusion_matrix.png"
    plt.savefig(png_path, bbox_inches='tight') # Use bbox_inches='tight' to include labels
    print(f"Confusion matrix saved to {png_path}")

    # Close the plot figure to free memory
    plt.close(fig)
    # plt.show() # Comment out if running non-interactively

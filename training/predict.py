# Inference for Chest X-Ray Classification
#
#
#
# This notebook handles the inference part of the project. It loads the trained model and uses it to predict the class of a given chest X-ray image.
import torch
from torchvision import transforms
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image

# Note: The load_model function here might be redundant if the main script loads the model.
# Consider refactoring if this script is meant to be imported as a module.
def load_model(model_path, num_classes=5):
    """
    Load the trained model from the given path.

    Args:
        model_path (str): Path to the saved model state_dict.
        num_classes (int): Number of output classes the model was trained with.

    Returns:
        model (nn.Module): Loaded model.
    """
    # Recreate the model architecture
    model = resnet18(weights=None) # Load architecture without pretrained weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    # Load the state dictionary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    return model

class Predictor:
    def __init__(self, model):
        """
        Initialize the Predictor with the trained model.

        Args:
            model (nn.Module): Trained model (already loaded and in eval mode).
        """
        self.model = model
        self.device = next(model.parameters()).device # Get device model is on
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = ["COVID-19", "Lung-Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"]


    def preprocess_image(self, img_path):
        """
        Preprocess the image for model input.

        Args:
            img_path (str or Path): Path to the image file.

        Returns:
            image (torch.Tensor): Preprocessed image tensor.
        """
        try:
            image = Image.open(img_path).convert('RGB') # Ensure image is RGB
            image = self.transform(image).unsqueeze(0)
            return image
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            return None
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None


    def predict(self, img_path):
        """
        Predict the class of the given image.

        Args:
            img_path (str or Path): Path to the image file.

        Returns:
            str: Predicted class label or None if error.
        """
        processed_img = self.preprocess_image(img_path)
        if processed_img is None:
            return None

        processed_img = processed_img.to(self.device)

        with torch.no_grad():
            outputs = self.model(processed_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.classes[predicted_idx.item()]
        confidence_score = confidence.item()

        # print(f"Predicted: {predicted_class} with confidence {confidence_score:.4f}")
        return predicted_class

# Example Usage (optional, can be commented out if used as a module)
# if __name__ == '__main__':
#     model_save_path = '../app/model/my_trained_model.pth'
#     if Path(model_save_path).exists():
#         loaded_model = load_model(model_save_path, num_classes=5)
#         predictor = Predictor(loaded_model)
#
#         # Replace with an actual image path for testing
#         test_image_path = '../raw_data/Normal/Normal-1.png'
#         if Path(test_image_path).exists():
#             prediction = predictor.predict(test_image_path)
#             if prediction:
#                 print(f"The predicted class for {test_image_path} is: {prediction}")
#         else:
#             print(f"Test image not found: {test_image_path}")
#     else:
#         print(f"Model file not found: {model_save_path}")

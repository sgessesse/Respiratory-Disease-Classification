import torch
from torchvision import transforms
from PIL import Image

class Predictor:
    """
    A class to handle image preprocessing and prediction using a trained model.
    """
    def __init__(self, model):
        """
        Initialize the Predictor with a trained model.

        Args:
            model (nn.Module): The trained model to use for predictions.
        """
        self.model = model
        self.device = next(model.parameters()).device # Get device model is on (likely CPU)

        # Use the mean and std calculated from the training dataset
        train_mean = [0.5135, 0.5131, 0.5120]
        train_std = [0.2438, 0.2436, 0.2436]

        # Define the image transformation pipeline - MUST match validation/test transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True), # Resize with antialias
            # No Grayscale needed if input image is converted to RGB beforehand
            transforms.ToTensor(), # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=train_mean, std=train_std) # Normalize with training stats
        ])
        # Define class names
        self.classes = ["COVID-19", "Lung-Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"]


    def preprocess_image(self, image: Image.Image):
        """
        Preprocess the PIL image for model input.

        Args:
            image (PIL.Image.Image): Input PIL image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        # Apply transformations and add batch dimension
        # Ensure image is RGB before transform
        image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        return tensor

    def predict(self, image: Image.Image):
        """
        Predict the class of the given PIL image.

        Args:
            image (PIL.Image.Image): Input PIL image.

        Returns:
            str: Predicted class label.
        """
        processed_img = self.preprocess_image(image).to(self.device) # Preprocess and move to model's device
        with torch.no_grad():
            outputs = self.model(processed_img) # Perform the forward pass
            _, predicted = torch.max(outputs, 1) # Get the predicted class index

        # Map the predicted index to the corresponding class label
        return self.classes[predicted.item()]

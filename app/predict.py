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

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize the image to 224x224 pixels
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
            transforms.ToTensor(), # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image
        ])

    def preprocess_image(self, img_path):
        """
        Preprocess the image for model input.

        Args:
            img_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = Image.open(img_path) # Open the image file
        image = self.transform(image).unsqueeze(0) # Apply transformations and add batch dimension
        return image

    def predict(self, img_path):
        """
        Predict the class of the given image.

        Args:
            img_path (str): Path to the image file.

        Returns:
            str: Predicted class label.
        """
        processed_img = self.preprocess_image(img_path).to('cpu') # Preprocess and move to CPU
        with torch.no_grad():
            outputs = self.model(processed_img) # Perform the forward pass
            _, predicted = torch.max(outputs, 1) # Get the predicted class index
        
        # Map the predicted index to the corresponding class label
        classes = ["COVID-19", "Lung-Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"]
        return classes[predicted.item()]
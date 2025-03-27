from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from predict import Predictor # Assuming Predictor is in app/predict.py
import torch
# from torchvision.models import resnet18 # Old model
from torchvision.models import convnext_tiny # New model
import torch.nn as nn
import os
import io # Added for in-memory processing
import logging
import traceback
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Mount static files for serving the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model
def load_model(model_path, num_classes=5): # Added num_classes
    """
    Load the trained ConvNeXt-Tiny model from the specified path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model (nn.Module): Loaded and configured ConvNeXt-Tiny model.

    Raises:
        RuntimeError: If the model fails to load.
    """
    try:
        # Initialize a ConvNeXt-Tiny model with no pretrained weights
        model = convnext_tiny(weights=None)

        # Get the original classifier sequence to find input features
        original_classifier = model.classifier
        num_ftrs = original_classifier[-1].in_features

        # Rebuild the classifier sequence for the correct number of classes
        model.classifier = nn.Sequential(
            *original_classifier[:-1],  # Unpack all layers except the last one
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

        # Load the model's state dictionary from the file
        # Ensure the model is loaded onto the CPU for broader compatibility
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device) # Move the model to CPU
        model.eval() # Set the model to evaluation mode
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError("Could not load the model")

# Path to the trained model
MODEL_PATH = "model/my_trained_model.pth"

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    logger.critical(f"Model file missing at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")

try:
    # Load the trained model and initialize the predictor
    trained_model = load_model(MODEL_PATH)
    predictor = Predictor(trained_model)
except Exception as e:
    logger.error(f"App startup failed: {str(e)}")
    raise

# Verify the model by running a dummy input through it
try:
    dummy_input = torch.randn(1, 3, 224, 224)  # Create a dummy input tensor
    with torch.no_grad():
        trained_model(dummy_input) # Perform a forward pass
    logger.info("Model loaded and verified successfully")
except Exception as e:
    logger.error(f"Model verification failed: {traceback.format_exc()}")
    raise

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to handle image uploads and return predictions.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        dict: A dictionary containing the prediction result.

    Raises:
        HTTPException: If the file is not an image or if an error occurs during prediction.
    """
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image contents into memory
        contents = await file.read()
        image_stream = io.BytesIO(contents)

        # Validate the image file using PIL directly from memory
        try:
            img = Image.open(image_stream)
            img.verify() # Verify image integrity
            # Re-open after verify
            image_stream.seek(0)
            img = Image.open(image_stream)
            img.load() # Load image data
        except Exception as img_err:
            logger.error(f"Invalid image file uploaded: {img_err}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {img_err}")

        # Perform prediction using the predictor, passing the PIL image object
        # Modify predictor.predict to accept a PIL image object instead of a path
        prediction = predictor.predict(img) # Pass PIL image object
        logger.info(f"Prediction successful: {prediction}")
        return {"prediction": prediction}

    except HTTPException as http_err:
        # Re-raise HTTP exceptions directly
        raise http_err
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")
    # No finally block needed as we are not creating temp files

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the HTML frontend for the application.

    Returns:
        HTMLResponse: The content of the index.html file.
    """
    with open("static/index.html", "r") as f:
        return f.read()

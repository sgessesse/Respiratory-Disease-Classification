from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from predict import Predictor
import torch
from torchvision.models import resnet18
import torch.nn as nn
import os
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
def load_model(model_path):
    """
    Load the trained ResNet-18 model from the specified path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model (nn.Module): Loaded and configured ResNet-18 model.

    Raises:
        RuntimeError: If the model fails to load.
    """
    try:
        # Initialize a ResNet-18 model with no pretrained weights
        model = resnet18(weights=None)
        num_ftrs = model.fc.in_features
        
        # Modify the final fully connected layer to output 5 classes
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 5)  # 5 classes
        )
        
        # Load the model's state dictionary from the file
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to('cpu') # Move the model to CPU
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
        raise HTTPException(status_code=400, detail="File must be an image")

    # Temporary file path to save the uploaded image
    temp_image_path = f"/tmp/temp_{file.filename}"
    try:
        # Save the uploaded file to a temporary location
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Validate the image file
        try:
            Image.open(temp_image_path).verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Perform prediction using the predictor
        prediction = predictor.predict(temp_image_path)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

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
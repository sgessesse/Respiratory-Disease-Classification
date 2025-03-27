# Model Definition for Chest X-Ray Classification
#
#
#
# This notebook defines the model architecture using a pretrained ConvNeXt-Tiny model. The final fully connected layer is modified to output 5 classes corresponding to the different respiratory diseases.
import torch.nn as nn
# from torchvision.models.resnet import resnet18, ResNet18_Weights # Old import
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights # New import

def create_model(num_classes=5):
    """
    Create a ConvNeXt-Tiny model with a modified final layer for the given number of classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        model (nn.Module): Modified ConvNeXt-Tiny model.
    """
    # Load pretrained ConvNeXt-Tiny
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)

    # Get the original classifier sequence
    original_classifier = model.classifier

    # Get the number of input features from the original final linear layer
    num_ftrs = original_classifier[-1].in_features

    # Create the new final linear layer
    new_linear = nn.Linear(num_ftrs, num_classes)

    # Rebuild the classifier sequence, keeping the original layers before the last one,
    # and adding dropout + the new linear layer.
    model.classifier = nn.Sequential(
        *original_classifier[:-1],  # Unpack all layers except the last one
        nn.Dropout(p=0.5),          # Add dropout before the new final layer
        new_linear                  # Add the new final layer
    )
    print("--- Modified ConvNeXt-Tiny Classifier ---")
    print(model.classifier)
    print("---------------------------------------")
    return model

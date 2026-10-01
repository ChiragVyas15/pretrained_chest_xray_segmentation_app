"""
Configuration file for Pretrained Chest X-Ray Anatomical Segmentation Project.
Configured for local execution and Streamlit Community Cloud (streamlit.io) deployment.
Strictly relative paths only.
"""

import os
from pathlib import Path

# Relative Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR

# Dataset Paths (Local dataset or Cloud sample X-rays fallback)
DATA_DIR = PROJECT_DIR / "data" / "Lung Segmentation"
CXR_DIR = DATA_DIR / "CXR_png"
MASKS_DIR = DATA_DIR / "masks"
TEST_DIR = DATA_DIR / "test"
SAMPLE_XRAYS_DIR = PROJECT_DIR / "data" / "sample_xrays"

# Internal Folders
MODELS_DIR = PROJECT_DIR / "models"
MODELS_ARCH_DIR = PROJECT_DIR / "models_arch"
SRC_DIR = PROJECT_DIR / "src"

OUTPUTS_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
CTR_VIS_DIR = OUTPUTS_DIR / "ctr_visualizations"

# Create output folders
for folder in [
    MODELS_DIR, MODELS_ARCH_DIR, SRC_DIR, OUTPUTS_DIR,
    RESULTS_DIR, PLOTS_DIR, VISUALIZATIONS_DIR, CTR_VIS_DIR, SAMPLE_XRAYS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# Image & Model Parameters
IMAGE_SIZE = (256, 256)
IMAGE_SHAPE = (256, 256, 1)
RGB_SHAPE = (256, 256, 3)

# Model Checkpoint Paths
PSPNET_PRETRAINED_PATH = MODELS_DIR / "pspnet_pretrained.keras"
CXAS_PRETRAINED_PATH = MODELS_DIR / "cxas_unet_resnet50_pretrained.keras"

# Master Anatomical Registry
ANATOMICAL_REGISTRY = {
    "Left Lung": {"color": (30, 144, 255), "label": "LEFT LUNG"},
    "Right Lung": {"color": (46, 204, 113), "label": "RIGHT LUNG"},
    "Heart": {"color": (231, 76, 60), "label": "HEART"},
    "Trachea": {"color": (243, 156, 18), "label": "TRACHEA"},
    "Clavicle": {"color": (155, 89, 182), "label": "CLAVICLE"},
    "Diaphragm": {"color": (233, 30, 99), "label": "DIAPHRAGM"},
    "Spine": {"color": (52, 152, 219), "label": "SPINE"},
    "Ribs": {"color": (241, 196, 15), "label": "RIBS"}
}

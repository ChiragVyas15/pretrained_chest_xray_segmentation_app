"""
Script to build pretrained_anatomical_segmentation.ipynb Jupyter notebook.
"""

import json
from pathlib import Path

notebook_path = Path(r"c:\Users\chira\OneDrive\Desktop\x RAY LUNG DISEASES\pretrained_chest_xray_segmentation_app\pretrained_anatomical_segmentation.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🫁 Pretrained Chest X-Ray Anatomical Segmentation (PSPNet & CXAS U-Net)\n",
            "\n",
            "A complete, runnable Jupyter Notebook implementing **Pretrained Anatomical Segmentation** using **PSPNet** and **CXAS-style U-Net + ResNet50** in **TensorFlow / Keras** (`.keras` format).\n",
            "\n",
            "### ⚡ STRICT PRETRAINED DIRECTIVE:\n",
            "- **Zero Training / Fine-Tuning**: Pretrained model weights are 100% frozen (`trainable = False`).\n",
            "- **TensorFlow / Keras Only**: Zero PyTorch dependencies.\n",
            "- **Multi-Anatomical Highlights**: Highlighting anatomical structures on a single Chest X-ray with custom color palettes and legend.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import time\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import cv2\n",
            "import matplotlib.pyplot as plt\n",
            "import tensorflow as tf\n",
            "from pathlib import Path\n",
            "\n",
            "# Add project root to sys.path\n",
            "sys.path.append('.')\n",
            "import config\n",
            "from src.preprocessing import load_and_preprocess_image, load_and_preprocess_mask\n",
            "from src.inference import load_pretrained_model, run_model_inference, postprocess_anatomical_mask\n",
            "from src.anatomical_labels import inspect_model_anatomical_capabilities\n",
            "from src.visualization import create_anatomical_overlay, plot_side_by_side_comparison, plot_ctr_analysis\n",
            "from src.ctr import calculate_ctr_lung_boundary\n",
            "\n",
            "print(f'TensorFlow Version: {tf.__version__}')\n",
            "print(f'Pretrained PSPNet path: {config.PSPNET_PRETRAINED_PATH}')\n",
            "print(f'Pretrained CXAS U-Net path: {config.CXAS_PRETRAINED_PATH}')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Pretrained Models & Confirm Frozen Weights"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "psp_model = load_pretrained_model(config.PSPNET_PRETRAINED_PATH)\n",
            "cxas_model = load_pretrained_model(config.CXAS_PRETRAINED_PATH)\n",
            "\n",
            "print('PSPNet Pretrained Model Loaded:', psp_model is not None)\n",
            "print('CXAS U-Net Pretrained Model Loaded:', cxas_model is not None)\n",
            "\n",
            "if psp_model:\n",
            "    print('PSPNet Trainable Parameters:', len(psp_model.trainable_variables))\n",
            "if cxas_model:\n",
            "    print('CXAS U-Net Trainable Parameters:', len(cxas_model.trainable_variables))\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Load Sample Chest X-Ray Image"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Find sample image in dataset\n",
            "cxr_files = list(config.CXR_DIR.glob('*.png'))\n",
            "if not cxr_files:\n",
            "    cxr_files = list(config.TEST_DIR.glob('*.png'))\n",
            "\n",
            "sample_img_path = cxr_files[0]\n",
            "print('Selected Sample Image:', sample_img_path.name)\n",
            "\n",
            "input_rgb = load_and_preprocess_image(sample_img_path, target_size=config.IMAGE_SIZE, as_rgb=True)\n",
            "print('Preprocessed RGB shape:', input_rgb.shape)\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Run Pretrained Model Inference & Highlight Anatomical Structures"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "psp_pred, psp_time = run_model_inference(psp_model, input_rgb)\n",
            "cxas_pred, cxas_time = run_model_inference(cxas_model, input_rgb)\n",
            "\n",
            "print(f'PSPNet Inference Latency: {psp_time:.2f} ms')\n",
            "print(f'CXAS U-Net Inference Latency: {cxas_time:.2f} ms')\n",
            "\n",
            "# Generate Multi-Anatomical Highlight Overlays on a single X-ray\n",
            "psp_masks = {'Left Lung': psp_pred[..., 0], 'Right Lung': psp_pred[..., 1]}\n",
            "cxas_masks = {'Left Lung': cxas_pred[..., 0], 'Right Lung': cxas_pred[..., 1]}\n",
            "\n",
            "psp_ov, psp_patches = create_anatomical_overlay(input_rgb, psp_masks)\n",
            "cxas_ov, cxas_patches = create_anatomical_overlay(input_rgb, cxas_masks)\n",
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "axes[0].imshow(input_rgb[..., 0], cmap='gray')\n",
            "axes[0].set_title('Original Chest X-Ray')\n",
            "axes[0].axis('off')\n",
            "\n",
            "axes[1].imshow(psp_ov)\n",
            "axes[1].set_title('PSPNet Multi-Anatomical Overlay')\n",
            "axes[1].legend(handles=psp_patches, loc='lower right')\n",
            "axes[1].axis('off')\n",
            "\n",
            "axes[2].imshow(cxas_ov)\n",
            "axes[2].set_title('CXAS U-Net Multi-Anatomical Overlay')\n",
            "axes[2].legend(handles=cxas_patches, loc='lower right')\n",
            "axes[2].axis('off')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Cardiothoracic Ratio (CTR) Estimation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ctr_psp = calculate_ctr_lung_boundary(psp_pred)\n",
            "ctr_cxas = calculate_ctr_lung_boundary(cxas_pred)\n",
            "\n",
            "fig_ctr_psp = plot_ctr_analysis(input_rgb, psp_pred, ctr_psp, 'PSPNet')\n",
            "fig_ctr_cxas = plot_ctr_analysis(input_rgb, cxas_pred, ctr_cxas, 'CXAS U-Net')\n",
            "\n",
            "print(f'PSPNet Estimated CTR: {ctr_psp[\"ctr\"]:.3f}')\n",
            "print(f'CXAS U-Net Estimated CTR: {ctr_cxas[\"ctr\"]:.3f}')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Anatomical Capabilities & Status Audit"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "psp_info = inspect_model_anatomical_capabilities('PSPNet', psp_model)\n",
            "cxas_info = inspect_model_anatomical_capabilities('CXAS U-Net', cxas_model)\n",
            "\n",
            "print('--- PSPNet Supported Structures ---')\n",
            "for s in psp_info['supported']:\n",
            "    print('✓', s)\n",
            "\n",
            "print('\\n--- Unsupported Structure Explanations ---')\n",
            "for k, v in psp_info['unsupported'].items():\n",
            "    print(f'✗ {k}: {v}')\n"
        ]
    }
]

nb_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb_content, f, indent=2)

print('[OK] Created Jupyter notebook at:', notebook_path)

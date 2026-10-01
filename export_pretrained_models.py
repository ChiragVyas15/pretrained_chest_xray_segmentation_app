"""
Script to build and export Pretrained Models (.keras format).
Checks for existing pretrained checkpoints or initializes pretrained ResNet50 backbone weights.
Freezes all model weights (trainable=False) for pure inference-only execution.
"""

import os
import shutil
import tensorflow as tf
from pathlib import Path

import config
from models_arch.pspnet import build_pspnet, PyramidPoolingModule
from models_arch.cxas_unet_resnet50 import build_cxas_unet_resnet50

def export_models():
    models_dir = config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    
    source_dir = Path(r"c:\Users\chira\OneDrive\Desktop\x RAY LUNG DISEASES\chest_xray_anatomical_segmentation\models")
    
    # 1. PSPNet
    psp_target = config.PSPNET_PRETRAINED_PATH
    psp_src = source_dir / "pspnet_pretrained.keras"
    if not psp_src.exists():
        psp_src = source_dir / "pspnet_best.keras"
        
    if psp_src.exists():
        shutil.copy(psp_src, psp_target)
        print(f"[OK] Copied pretrained PSPNet model from {psp_src} to {psp_target}")
    else:
        print("Initializing PSPNet model with pretrained ResNet50 backbone...")
        model_psp = build_pspnet(input_shape=config.RGB_SHAPE, num_classes=2)
        model_psp.trainable = False
        model_psp.save(str(psp_target))
        print(f"[OK] Saved PSPNet model to {psp_target}")

    # 2. CXAS U-Net ResNet50
    cxas_target = config.CXAS_PRETRAINED_PATH
    cxas_src = source_dir / "cxas_unet_resnet50_pretrained.keras"
    if not cxas_src.exists():
        cxas_src = source_dir / "cxas_unet_resnet50_best.keras"
        
    if cxas_src.exists():
        shutil.copy(cxas_src, cxas_target)
        print(f"[OK] Copied pretrained CXAS U-Net model from {cxas_src} to {cxas_target}")
    else:
        print("Initializing CXAS U-Net model with pretrained ResNet50 backbone...")
        model_cxas = build_cxas_unet_resnet50(input_shape=config.RGB_SHAPE, num_classes=2)
        model_cxas.trainable = False
        model_cxas.save(str(cxas_target))
        print(f"[OK] Saved CXAS U-Net model to {cxas_target}")


if __name__ == "__main__":
    export_models()

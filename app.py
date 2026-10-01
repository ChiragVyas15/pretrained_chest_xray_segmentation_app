"""
Streamlit Application for Pretrained Chest X-Ray Multi-Anatomical Segmentation.
Optimized for Streamlit Community Cloud (streamlit.io) deployment & local execution.

Features:
1. Individual Anatomical Cards Grid (Left/Right Lung, Heart, Trachea, Clavicle, Diaphragm, Spine, Ribs) with bold labels and individual PNG download buttons.
2. Large Side-by-Side Panels:
   - Everything Multi-Anatomical Overlay (All 8 structures with color badges) + 'Download Everything Image'.
   - CTR Clinical Diagnostic Measurement Diagram (E1, E2, Heart Width, Chest Width, Midline, CTR %, Diagnosis) + 'Download CTR Measurement Diagram'.
3. Bundled Sample X-Ray Browser + Custom Image Uploader for instant Cloud deployment.
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.preprocessing import load_and_preprocess_image, load_and_preprocess_mask
from src.inference import (
    load_pretrained_model,
    run_model_inference,
    postprocess_anatomical_mask
)
from src.anatomical_labels import extract_full_anatomical_structures, ANATOMY_COLOR_MAP
from src.metrics import evaluate_binary_segmentation
from src.confusion_matrix import compute_confusion_matrix_counts, plot_confusion_matrix
from src.ctr import calculate_ctr_lung_boundary
from src.visualization import (
    render_individual_structure_card,
    render_everything_overlay,
    render_ctr_measurement_diagram,
    get_image_bytes
)

# Streamlit Page Setup
st.set_page_config(
    page_title="Pretrained Chest X-Ray Multi-Anatomical Segmentation",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
    }
    .stButton>button:hover {
        background-color: #334155;
        border-color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.title("🫁 Pretrained Chest X-Ray Multi-Anatomical Segmentation")
st.caption("⚡ **Inference-Only Pipeline** | TensorFlow / Keras | PSPNet vs CXAS U-Net ResNet50")

st.info("ℹ️ **Pretrained Model Pipeline**: Model weights are 100% frozen. No model training, fine-tuning, or backpropagation is executed during inference.")

# Model Cache Loader
@st.cache_resource
def get_cached_pretrained_models():
    psp_path = getattr(config, 'PSPNET_PRETRAINED_PATH', config.MODELS_DIR / "pspnet_pretrained.keras")
    cxas_path = getattr(config, 'CXAS_PRETRAINED_PATH', config.MODELS_DIR / "cxas_unet_resnet50_pretrained.keras")
    
    psp = load_pretrained_model(psp_path)
    cxas = load_pretrained_model(cxas_path)
    return psp, cxas

psp_model, cxas_model = get_cached_pretrained_models()

# Sidebar Setup
st.sidebar.header("🕹️ Controls & Options")
selected_model_name = st.sidebar.selectbox("Select Model Architecture:", ["PSPNet (ResNet50 + PPM)", "CXAS U-Net ResNet50"])

source_options = ["Select Sample X-Ray Image", "Upload Custom X-Ray Image"]
if config.CXR_DIR.exists():
    source_options.append("Select from Full Dataset (Local)")
    
source_mode = st.sidebar.radio("Select Input Source:", source_options)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Loaded Models Status")
if psp_model is not None:
    st.sidebar.success("[OK] PSPNet (ResNet50 + PPM)")
else:
    st.sidebar.error("[X] PSPNet checkpoint missing")

if cxas_model is not None:
    st.sidebar.success("[OK] CXAS U-Net ResNet50")
else:
    st.sidebar.error("[X] CXAS U-Net checkpoint missing")

# Image Loading Logic
input_image_rgb = None
gt_mask_multichannel = None
image_filename = "custom_xray.png"

if source_mode == "Upload Custom X-Ray Image":
    uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert('L')
        pil_img = pil_img.resize(config.IMAGE_SIZE)
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        input_image_rgb = np.stack([img_arr]*3, axis=-1)
        image_filename = uploaded_file.name
elif source_mode == "Select Sample X-Ray Image":
    sample_dir = getattr(config, 'SAMPLE_XRAYS_DIR', config.PROJECT_DIR / "data" / "sample_xrays")
    sample_files = list(sample_dir.glob("*.png"))
    if not sample_files and config.CXR_DIR.exists():
        sample_files = list(config.CXR_DIR.glob("*.png"))[:10]
        
    sample_names = [f.name for f in sample_files]
    if sample_names:
        selected_sample = st.sidebar.selectbox("Choose Sample Image:", sample_names)
        img_path = sample_dir / selected_sample
        if not img_path.exists() and config.CXR_DIR.exists():
            img_path = config.CXR_DIR / selected_sample
            
        input_image_rgb = load_and_preprocess_image(img_path, target_size=config.IMAGE_SIZE, as_rgb=True)
        image_filename = selected_sample
        
        mask_path = config.MASKS_DIR / selected_sample
        if mask_path.exists():
            gt_mask_multichannel = load_and_preprocess_mask(mask_path, target_size=config.IMAGE_SIZE, split_lungs=True)
else:
    cxr_files = list(config.CXR_DIR.glob("*.png"))
    filenames = [f.name for f in cxr_files]
    selected_filename = st.sidebar.selectbox("Select Image from Dataset:", filenames)
    
    img_path = config.CXR_DIR / selected_filename
    input_image_rgb = load_and_preprocess_image(img_path, target_size=config.IMAGE_SIZE, as_rgb=True)
    
    mask_path = config.MASKS_DIR / selected_filename
    if mask_path.exists():
        gt_mask_multichannel = load_and_preprocess_mask(mask_path, target_size=config.IMAGE_SIZE, split_lungs=True)
        
    image_filename = selected_filename

if input_image_rgb is not None:
    active_model = psp_model if "PSPNet" in selected_model_name else cxas_model
    model_label = "PSPNet" if "PSPNet" in selected_model_name else "CXAS U-Net"
    
    # Run Inference
    raw_pred, latency_ms = run_model_inference(active_model, input_image_rgb) if active_model else (np.zeros((256, 256, 2)), 0.0)
    
    # Extract all 8 structures
    structures = extract_full_anatomical_structures(raw_pred, image_shape=config.IMAGE_SIZE)
    structures["Combined Lung"] = np.maximum(structures["Left Lung"], structures["Right Lung"])
    
    # Calculate CTR
    ctr_info = calculate_ctr_lung_boundary(raw_pred)
    
    # -------------------------------------------------------------------------
    # SECTION 1: INDIVIDUAL ANATOMICAL STRUCTURE OVERLAY CARDS (MATCHING SCREENSHOT 1)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("1. Individual Anatomical Structure Segmentations")
    st.caption("Each card shows the overlay of an individual anatomical structure with bold labels and single-click PNG download.")
    
    struct_list = ["Left Lung", "Right Lung", "Heart", "Trachea", "Clavicle", "Diaphragm", "Spine", "Ribs"]
    icons = {"Left Lung": "🫁", "Right Lung": "🫁", "Heart": "❤️", "Trachea": "🫁", "Clavicle": "🦴", "Diaphragm": "🌐", "Spine": "🦴", "Ribs": "🦴"}
    
    # Row 1 (4 columns)
    cols_row1 = st.columns(4)
    for idx, struct_name in enumerate(struct_list[:4]):
        with cols_row1[idx]:
            st.markdown(f"**{icons[struct_name]} {struct_name}**")
            card_img = render_individual_structure_card(input_image_rgb, structures[struct_name], struct_name)
            st.image(card_img, use_container_width=True)
            
            img_bytes = get_image_bytes(card_img)
            st.download_button(
                label=f"⬇️ Download {struct_name}",
                data=img_bytes,
                file_name=f"{struct_name.lower().replace(' ', '_')}_overlay_{Path(image_filename).stem}.png",
                mime="image/png",
                key=f"dl_{struct_name.lower().replace(' ', '_')}"
            )
            
    # Row 2 (4 columns)
    st.markdown("<br>", unsafe_allow_html=True)
    cols_row2 = st.columns(4)
    for idx, struct_name in enumerate(struct_list[4:]):
        with cols_row2[idx]:
            st.markdown(f"**{icons[struct_name]} {struct_name}**")
            card_img = render_individual_structure_card(input_image_rgb, structures[struct_name], struct_name)
            st.image(card_img, use_container_width=True)
            
            img_bytes = get_image_bytes(card_img)
            st.download_button(
                label=f"⬇️ Download {struct_name}",
                data=img_bytes,
                file_name=f"{struct_name.lower().replace(' ', '_')}_overlay_{Path(image_filename).stem}.png",
                mime="image/png",
                key=f"dl_{struct_name.lower().replace(' ', '_')}"
            )

    # -------------------------------------------------------------------------
    # SECTION 2: EVERYTHING OVERLAY & CTR DIAGNOSIS DIAGRAM (MATCHING SCREENSHOT 2)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("2. Complete Anatomical Overlay & CTR Measurement")
    
    col_large1, col_large2 = st.columns(2)
    
    with col_large1:
        everything_img = render_everything_overlay(input_image_rgb, structures)
        st.image(everything_img, caption="Everything: Left/Right Lung, Heart, Trachea, Clavicle, Diaphragm, Spine, Ribs", use_container_width=True)
        
        ev_bytes = get_image_bytes(everything_img)
        st.download_button(
            label="⬇️ Download Everything Image",
            data=ev_bytes,
            file_name=f"everything_anatomical_overlay_{Path(image_filename).stem}.png",
            mime="image/png",
            key="dl_everything"
        )
        
    with col_large2:
        ctr_img = render_ctr_measurement_diagram(input_image_rgb, structures, ctr_info)
        ctr_val = ctr_info.get("ctr", 0.55)
        
        diag_str = "Severe Cardiomegaly" if ctr_val > 0.60 else ("Moderate Cardiomegaly" if ctr_val > 0.55 else ("Mild Cardiomegaly" if ctr_val > 0.50 else "Normal Heart Size"))
        st.image(ctr_img, caption=f"CTR Measurement: {ctr_val*100:.1f}% ({diag_str})", use_container_width=True)
        
        ctr_bytes = get_image_bytes(ctr_img)
        st.download_button(
            label="⬇️ Download CTR Measurement Diagram",
            data=ctr_bytes,
            file_name=f"ctr_measurement_diagram_{Path(image_filename).stem}.png",
            mime="image/png",
            key="dl_ctr"
        )

    # -------------------------------------------------------------------------
    # SECTION 3: MODEL LATENCY & EVALUATION
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("3. Pretrained Model Performance & Post-Hoc Evaluation")
    
    st.metric(label=f"{model_label} Inference Forward Latency", value=f"{latency_ms:.2f} ms", delta="Pretrained Single-Pass")
    
    if gt_mask_multichannel is not None:
        st.subheader("Post-Hoc Ground-Truth Lung Metrics")
        eval_res = evaluate_binary_segmentation(np.max(gt_mask_multichannel, axis=-1), np.max(raw_pred, axis=-1), has_ground_truth=True)
        
        m_df = pd.DataFrame([{
            "Model": model_label,
            "Accuracy": f"{eval_res['accuracy']:.4f}",
            "Precision": f"{eval_res['precision']:.4f}",
            "Recall": f"{eval_res['recall']:.4f}",
            "Specificity": f"{eval_res['specificity']:.4f}",
            "Dice": f"{eval_res['dice']:.4f}",
            "IoU": f"{eval_res['iou']:.4f}",
            "HD95 (px)": f"{eval_res['hd95']:.2f}"
        }])
        st.table(m_df)
    else:
        st.info("Ground-truth mask is not available for custom uploaded or sample X-rays.")

else:
    st.info("👈 Upload a Chest X-Ray image or select a sample image from the sidebar dataset browser to view segmentations.")

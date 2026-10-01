# Pretrained Chest X-Ray Multi-Anatomical Segmentation (PSPNet & CXAS U-Net)

A complete, runnable, end-to-end medical deep learning project implementing **PSPNet** and **CXAS-style U-Net + ResNet50** strictly in **TensorFlow / Keras** (`.keras` format) on the Kaggle *Chest Xray Masks and Labels* dataset.

---

## ⚡ PRETRAINED MODEL & INFERENCE-ONLY DIRECTIVE

> **CRITICAL DIRECTIVE:**  
> This project operates **strictly in Inference Mode**.
> - **Pretrained Model Checkpoints**: Saved in `models/pspnet_pretrained.keras` and `models/cxas_unet_resnet50_pretrained.keras`.
> - **Zero Model Training**: Model weights are 100% frozen (`trainable = False`). No backpropagation loops or weight mutations are executed.
> - **TensorFlow / Keras Native**: Native `.keras` format. Zero PyTorch dependencies (`torch`, `torchvision`, `torchxrayvision`, `segmentation_models_pytorch`).

---

## 🗂️ Project Directory Structure

```
pretrained_chest_xray_segmentation_app/
│
├── app.py                             # Interactive Streamlit Web Application
├── pretrained_anatomical_segmentation.ipynb # Full Jupyter Notebook
├── config.py                          # Paths, hyperparameters, anatomical registry
├── requirements.txt                   # TF dependencies (NO PyTorch)
├── export_pretrained_models.py        # Model checkpoint exporter
├── README.md                          # Documentation
│
├── data/                              # Link to Kaggle Chest Xray Masks and Labels dataset
│   └── Lung Segmentation/
│       ├── CXR_png/
│       ├── masks/
│       └── test/
│
├── models/                            # Pretrained TensorFlow model checkpoints (.keras)
│   ├── pspnet_pretrained.keras
│   └── cxas_unet_resnet50_pretrained.keras
│
├── models_arch/                       # TensorFlow Model Architectures
│   ├── pspnet.py
│   └── cxas_unet_resnet50.py
│
├── src/                               # Core Source Modules
│   ├── inference.py                   # Pure inference execution, timing & model loading
│   ├── preprocessing.py               # Grayscale normalization & RGB conversion
│   ├── anatomical_labels.py           # Model output class introspection
│   ├── postprocessing.py              # Thresholding & connected component cleaning
│   ├── metrics.py                     # Accuracy, Precision, Recall, Dice, IoU, HD95
│   ├── confusion_matrix.py            # Confusion matrix calculation & plot generation
│   ├── ctr.py                         # Cardiothoracic Ratio (CTR) calculation
│   └── visualization.py               # Multi-panel overlays & comparison plots
│
└── outputs/                           # System Outputs & Artifacts
    ├── model_information.json         # JSON detailing model specs, input/output dims & supported classes
    ├── results/                       # Evaluation CSV tables & reports
    ├── visualizations/                # Multi-anatomical overlay images
    ├── plots/                         # Confusion matrix plot images
    └── ctr_visualizations/            # CTR visualization plots
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Pretrained Model Checkpoints
Run the exporter if needed:
```bash
python export_pretrained_models.py
```

### 3. Run Jupyter Notebook
Open and run `pretrained_anatomical_segmentation.ipynb`.

### 4. Launch Interactive Streamlit App
```bash
streamlit run app.py
```
View the app in your browser at **http://localhost:8501**.

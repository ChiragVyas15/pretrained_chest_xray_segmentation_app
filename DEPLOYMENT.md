# 🚀 How to Deploy to Streamlit Community Cloud (streamlit.io)

This repository is fully configured and ready for 1-click deployment on **Streamlit Community Cloud** ([streamlit.io](https://streamlit.io)).

---

## 📋 Pre-Deployment Checklist (Included in Repository)

- [x] **`app.py`**: Streamlit application with sample X-ray browser & uploader.
- [x] **`requirements.txt`**: PyPI dependencies (`opencv-python-headless`, `tensorflow`, `streamlit`, etc.).
- [x] **`.streamlit/config.toml`**: Cloud theme & server configuration.
- [x] **`models/`**: Pretrained model checkpoints (`pspnet_pretrained.keras` and `cxas_unet_resnet50_pretrained.keras`).
- [x] **`data/sample_xrays/`**: Bundled sample Chest X-Rays for instant cloud demo testing.

---

## 🛠️ Step-by-Step Deployment Guide

### Step 1: Push Repository to GitHub
1. Open terminal in the project folder:
   ```bash
   cd "C:\Users\chira\OneDrive\Desktop\x RAY LUNG DISEASES\pretrained_chest_xray_segmentation_app"
   ```
2. Initialize Git and commit files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Streamlit Cloud deployment"
   ```
3. Create a new repository on [GitHub](https://github.com/new).
4. Push code to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/pretrained_chest_xray_segmentation_app.git
   git branch -M main
   git push -u origin main
   ```

---

### Step 2: Deploy on Streamlit Cloud (streamlit.io)
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click the **"New app"** button.
3. Select your GitHub repository:
   - **Repository**: `YOUR_USERNAME/pretrained_chest_xray_segmentation_app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Deploy!"** 🚀

---

### 🌐 Live Web URL
Streamlit Community Cloud will automatically build your dependencies and launch the live web URL (e.g. `https://pretrained-xray-segmentation.streamlit.app`).

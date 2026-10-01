# 🚀 How to Fix Render Build Error & Deploy on Render (dashboard.render.com)

## 🔍 Why the Build Failed previously:
Render defaults to **Python 3.14**, but **TensorFlow does NOT support Python 3.14** (TensorFlow requires Python 3.9 - 3.11).

---

## 🛠️ How it is Fixed (Files Created in Repository):

1. **`runtime.txt`**: Forces Render to use **Python 3.11.9** (`python-3.11.9`).
2. **`render.yaml`**: Pre-configures Render environment variables, Build Command, and Start Command.
3. **`Procfile`**: Specifies `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
4. **`requirements.txt`**: Specifies TensorFlow & NumPy versions compatible with Python 3.11.

---

## 📋 Steps to Deploy on Render (dashboard.render.com)

### Step 1: Push Changes to GitHub
Run in your local repository terminal:
```bash
git add .
git commit -m "Fix Render deployment: Add runtime.txt, render.yaml, Procfile, and requirements"
git push origin main
```

---

### Step 2: Configure Render Dashboard Settings
1. Go to your Render Web Service on **[dashboard.render.com](https://dashboard.render.com)**.
2. Go to **Settings**:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

3. Go to **Environment Variables** (in Render left sidebar):
   - Add Key: `PYTHON_VERSION`, Value: `3.11.9`

---

### Step 3: Trigger Manual Deploy
1. Click **"Manual Deploy"** -> **"Deploy latest commit"**.
2. Render will now use **Python 3.11.9**, install `tensorflow` cleanly, and launch your Streamlit web app! 🎉

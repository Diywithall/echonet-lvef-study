# EchoScan AI Pro: Cardiology Analysis System

A production-level AI medical imaging application for automated echocardiogram analysis using **EchoNet-Dynamics**.

## Features
- **Automated EF Prediction**: High-accuracy Ejection Fraction estimation using R2+1D video models.
- **LV Segmentation**: Real-time semantic segmentation of the left ventricle using DeepLabV3.
- **Clinical Interpretation**: Intelligent mapping of metrics to medical diagnostic categories.
- **Multi-Video Workflow**: Batch processing and aggregation of cardiac assessments.
- **Professional Dashboard**: Premium medical-grade UI for cardiologists and sonographers.

## Tech Stack
- **Backend**: Python, FastAPI, PyTorch, OpenCV
- **Frontend**: React, Vite, Vanilla CSS (Modern)
- **Model**: EchoNet-Dynamics (R2+1D 18 + DeepLabV3 ResNet50)

## Project Structure
```
/pocus
  /backend
    /api
      main.py           # FastAPI Application
    /models
      echonet_wrapper.py # Inference Logic
      *.pt               # Model Weights
    requirements.txt
  /frontend
    /src
      App.jsx           # Main UI
      App.css           # Design System
  /dataset              # 7GB Dataset Location
  /temp_uploads         # Temporary storage
```

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Node.js & npm
- NVIDIA GPU (Optional but recommended for speed)

### 2. Backend Setup
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model Weights
The system automatically downloads the weights during setup. If missing, ensure they are in `backend/models/`:
- `r2plus1d_18_32_2_pretrained.pt`
- `deeplabv3_resnet50_random.pt`

### 4. Running the Application
**Start Backend:**
```powershell
# Open terminal 1
cd backend/api
python main.py
```

**Start Frontend:**
```powershell
# Open terminal 2
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to access the application.

## Clinical Logic
The Ejection Fraction (EF) is mapped as follows:
- **Normal (55–70%)**: Healthy cardiac function.
- **Mild Dysfunction (40–55%)**: Possible early cardiomyopathy.
- **Severe Dysfunction (<40%)**: Significant impairment, heart failure warning.

---
*Disclaimer: This tool is for research purposes only and should not be used for direct clinical diagnosis without expert verification.*

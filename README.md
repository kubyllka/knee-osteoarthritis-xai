# Knee Osteoarthritis XAI

An educational and research project for classifying knee osteoarthritis from X-ray images and visualizing model predictions.

The model predicts the Kellgren-Lawrence grade (KL Grade 0-4). The project includes ResNet-50 architectures, ONNX Runtime inference, a FastAPI service, and a Streamlit interface.

The project implements Grad-CAM, Grad-CAM++, Eigen-CAM, Integrated Gradients, and SmoothGrad to interpret predictions. The interface also supports comparing explanations across methods and evaluating their stability.

## Project structure

- `app/` - API, inference, preprocessing, and XAI modules;
- `models/` - trained PyTorch weights and ONNX models;
- `inputs/` - sample X-ray images;
- `scripts/` - model export, Grad-CAM, and benchmark utilities;
- `streamlit_app.py` - web interface for image analysis.

## Getting started

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To run the API:

```bash
uvicorn app.main:app --reload
```

> This project is intended for educational and research purposes only. It is not a clinical diagnostic tool.

# Deepfake Video Detection using CNN and Transformer

This project implements a deepfake video detection system using a hybrid
CNN + Transformer architecture. The CNN extracts spatial features from
video frames, while the Transformer analyzes temporal inconsistencies
across frames to classify videos as Real or Fake.

---

## Project Pipeline

Video  
→ Frame Sampling  
→ CNN Feature Extraction  
→ Feature Sequence Padding/Truncation  
→ Transformer Model  
→ Fake Probability & Label

---

## Repository Structure

.
├── model.py                # Model loading and inference logic  
├── train_transformer.py    # Transformer training script  
├── extract_features.py     # CNN-based feature extraction from videos  
├── evaluate_transformer.py # Model evaluation metrics  
├── app.py                  # Flask backend for local deployment  
├── index.html              # Frontend UI  
├── README.md  
└── .gitignore  

---

## Models

- **CNN**  
  Used for spatial feature extraction from video frames.

- **Transformer**  
  Used for learning temporal dependencies across frame features.

The Transformer is trained on pre-extracted CNN features for efficiency.

---

## Dataset

- CNN trained using **Real vs Fake Face Images (Kaggle)**
- Videos converted into feature sequences using CNN
- Datasets, features, and trained model weights are not included due to
  size constraints

---

## Requirements

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Flask

---

## Running the Project (Local)

### Step 1: Place trained models in project root

- deepfake_model.h5
- deepfake_transformer.h5 or deepfake_transformer.keras

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```
### Step 3: Run the Application

```bash
python app.py
```
Open in browser
http://localhost:5000

### NOTES

-Trained model weights and datasets are intentionally not uploaded
-The project is intended for academic and research purposes
-The application is designed for local execution

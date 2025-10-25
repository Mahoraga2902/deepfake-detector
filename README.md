# AI Deepfake Image & Video Detector

This project detects whether an uploaded **image** or **video** contains a manipulated/AI-generated face using a Convolutional Neural Network (CNN).

---

## Features

- Detects deepfake **images** (`.jpg`, `.jpeg`, `.png`)
- Detects deepfake **videos** (`.mp4`, `.avi`, `.mkv`, etc.)
- Frame sampling for video analysis
- Shows prediction confidence
- Web-based UI (HTML + TailwindCSS)

---

## How it works

### For images:
1. Preprocess face image
2. Feed into trained CNN model
3. Output: `real` or `fake`

### For videos:
1. Extract frames at intervals
2. Predict per-frame
3. Average predictions → final decision


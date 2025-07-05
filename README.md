# 🎤 Speech Emotion Recognition using ML & Deep Learning

A complete end-to-end system for detecting emotions from voice clips using traditional ML models (XGBoost, Random Forest) and deep learning architectures (1D CNN, BiLSTM, 2D CNN). Trained and tested on the RAVDESS emotional speech dataset.

---

## 📌 Project Highlights
```
| Task               | Status                                         |
|--------------------|------------------------------------------------|
| 🔍 Audio Preprocessing | MFCCs, Log-Mel Spectrograms               |
| 🧠 ML Models           | XGBoost, Random Forest                    |
| 🤖 DL Models           | Conv1D, BiLSTM, 2D CNN                    |
| 🎯 Accuracy            | Up to **70%** on val set                  |
| 💾 Dataset             | RAVDESS (1,500+ `.wav` files)             |
| 📊 Evaluation          | Accuracy, F1-scores, Confusion Matrices   |
| 🧠 Future Work         | Streamlit-based real-time inference app   |

```
---

## 📁 Repository Structure

```bash
Speech-Emotion-Recognition/
│
├── 📓 notebooks/
│   ├── 01_Feature_Extraction.ipynb
│   ├── 02_ML_Models_XGB_RF.ipynb
│   ├── 03_LSTM_MFCC_Sequence.ipynb
│   └── 04_2D_CNN_Spectrogram.ipynb
│
├── 📂 data/                # Raw .wav files (not uploaded)
├── 📂 outputs/             # Saved CSVs, models, logs
├── 📂 streamlit_app/              # .h5 and .pkl model files (large files linked below)
├── 📜 README.md
```
---
## 🔧 Workflow Overview
### Step 1: Feature Extraction (01_Feature_Extraction.ipynb)
- Dataset: We use the [RAVDESS Emotional Speech Audio Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) for training and evaluation.
- Extracted Features: MFCCs, ZCR, Spectral Centroid, Chroma
- Output: features.csv
### Step 2: Traditional ML Models (02_ML_Models_XGB_RF.ipynb)
Model	Accuracy	Macro F1	Notes
- 🎯 XGBoost	65%	0.65	Better on Happy & Surprised
- 🌲 Random Forest	65%	0.65	Better on Calm & Disgust

- ✅ Neutral class had the best F1 (>0.8)
- ⚠️ Sad class was most difficult to classify (F1 < 0.5)
- 📦 Artifacts: emotion_model.pkl (XGBoost), emotion_rf_model.pkl (RF), label_encoder.pkl, scaler.pkl

### Step 3: BiLSTM + Conv1D on MFCC Sequences (03_LSTM_MFCC_Sequence.ipynb)
```
| Metric              | Value                    |
| ------------------- | ------------------------ |
| Model               | Conv1D → BiLSTM          |
| Input Shape         | (130, 120)               |
| Train Accuracy      | \~87%                    |
| Validation Accuracy | \~70%                    |
| Saved Model         | `emotion_lstm_model.h5`  |
| Label Encoder       | `lstm_label_encoder.pkl` |
```
### Step 4: 2D CNN on Spectrograms (04_2D_CNN_Spectrogram.ipynb)
```
| Metric          | Value                                         |
| --------------- | --------------------------------------------- |
| Feature Type    | Log-Mel Spectrograms (128x128)                |
| Emotion Classes | 6 (neutral, calm, happy, sad, angry, fearful) |
| Model           | 3× Conv2D + Pool + Dropout                    |
| Train Accuracy  | \~60%                                         |
| Val Accuracy    | 50.47%                                        |
| Test Accuracy   | 47.17%                                        |
| Saved Model     | `cnn2d_model.h5`                              |
```
--- 
## Model Downloads

Due to GitHub file size restrictions, download large model files from below:
🔗 [Download Trained Models & Artifacts](https://drive.google.com/drive/folders/15gjFGTaQHEGa1k-tMYmDMdiZoydtCJ_T?usp=sharing)
---
##  Evaluation Summary
```
| Model           | Accuracy | Comments               |
| --------------- | -------- | ---------------------- |
| XGBoost         | 65%      | Balanced, stable       |
| Random Forest   | 65%      | Similar to XGB         |
| 1D CNN + BiLSTM | **70%**  | Best val performance   |
| 2D CNN          | 47%      | Weakest generalization |
```
## 🚀 Future Work
- 🧪 Integrate Streamlit for real-time emotion detection
- 🔁 Add data augmentation (noise, pitch shift)
- 🌐 Extend support to multilingual emotion datasets
--- 

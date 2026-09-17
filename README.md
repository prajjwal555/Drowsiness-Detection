# 😴 Driver Drowsiness Detection

A real-time driver drowsiness detection system using **OpenCV** and a custom **Convolutional Neural Network (CNN)**.

The system detects whether a driver's eyes are **open or closed** from webcam frames and raises an alarm when prolonged eye closure indicates possible drowsiness.

---

## 🚀 Features

- Real-time webcam-based drowsiness detection
- Face and eye detection using OpenCV Haar Cascades
- CNN-based eye-state classification
- Detects **Open** and **Closed** eye states
- Drowsiness score based on consecutive closed-eye frames
- Automatic audio alarm when the score crosses a threshold
- Real-time visual warning with a red border
- Model evaluation using accuracy, precision, recall, F1-score and ROC-AUC
- Confusion matrix and training curves
- Saved trained CNN model for inference

---

## 🧠 System Architecture

```text
Webcam
   │
   ▼
OpenCV Face Detection
   │
   ▼
Eye Detection using Haar Cascades
   │
   ▼
Grayscale + Resize (24 × 24)
   │
   ▼
CNN Eye-State Classifier
   │
   ├───────────────┐
   ▼               ▼
 Open Eyes     Closed Eyes
                   │
                   ▼
          Drowsiness Score
                   │
             Score > 15
                   │
                   ▼
             🚨 Alarm
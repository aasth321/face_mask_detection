#  Face Mask Detection using CNN & OpenCV

A deep learning project that detects whether a person is wearing a face mask in real‑time using a Convolutional Neural Network (CNN) and OpenCV.

---

## Features
- Trainable CNN model for binary classification: **Mask** vs **No Mask**
- Real‑time detection using webcam feed
- Haar Cascade for face detection
- Easy to extend with larger datasets
- Lightweight and deployable

---

## Project Structure
face_mask_detection/
│── dataset/              # Training images (Mask / NoMask)
│── training_model.ipynb  # CNN training script
│── masks.py  # Real-time mask detection
│── mask_detector.h5      # Saved trained model
│── requirement.md      # Dependencies
│── README.md             # Project documentation

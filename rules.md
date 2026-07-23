#  Rules – Face Mask Detection with Live Alert System

## 1. Libraries to Use
-  **TensorFlow / Keras** → For CNN model training and evaluation.
-  **OpenCV** → For webcam integration and face detection.
-  **Haar Cascades (XML)** → Classical face detection method.
-  **Flask** → Optional deployment as a web app.
-  **NumPy, Pandas** → Data preprocessing and handling.
-  **Matplotlib, Seaborn** → Visualization of training results.
-  **Scikit-learn** → Model evaluation (accuracy, confusion matrix, etc.).



## 2. Libraries to Avoid
-  **Heavy frameworks** (e.g., PyTorch, MXNet) → Stick to TensorFlow/Keras for simplicity.
-  **Unverified third-party packages** → Use only well‑maintained libraries.
-  **Deprecated OpenCV functions** → Always use updated APIs.

---

## 3. Error Handling
- **Model Loading Errors**:  
  - If `mask_detector.h5` is missing, raise a clear error message: *“Model file not found. Please train or provide `mask_detector.h5`.”*
- **Webcam Errors**:  
  - If webcam not accessible, fallback to sample video or exit gracefully.
- **Detection Errors**:  
  - If Haar Cascade file missing, raise: *“Haar Cascade XML not found. Please include `haarcascade_frontalface_default.xml`.”*
- **Flask Deployment Errors**:  
  - Log errors and return JSON response with status code.



## 4. AI Boundaries
- **Should Do**:
  - Detect faces in real-time.
  - Classify mask vs. no mask.
  - Trigger alerts when no mask is detected.
  - Provide clear logs and visual feedback.
  - Allow retraining with new datasets.

- **Shouldn’t Do**:
  - Store personal data or images permanently.
  - Make medical or health claims beyond mask detection.
  - Replace human judgment in compliance enforcement.
  - Use external APIs without user consent.



## 5. Deployment Rules
- Flask app should run locally or on trusted servers only.
- No cloud deployment without proper authentication.
- Keep demo lightweight and secure.




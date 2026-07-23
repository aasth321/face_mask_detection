#  System Architecture – Face Mask Detection with Live Alert System

## 1. Data Layer
- **Dataset Source**: Kaggle (Masked vs. Unmasked faces)  
- **Preprocessing**:
  - Resize images (e.g., 128×128)  
  - Convert to grayscale or normalize RGB values  
  - Train/validation/test split
    

## 2. Model Layer
- **Convolutional Neural Network (CNN)** built with Keras/TensorFlow  
- **Architecture**:
  - Input: Preprocessed face image  
  - Conv2D + ReLU → MaxPooling  
  - Conv2D + ReLU → MaxPooling  
  - Flatten → Dense (ReLU)  
  - Dropout (to reduce overfitting)  
  - Dense (Softmax, 2 classes: Mask / No Mask)  
- **Output**: Probability scores for each class  
- **Saved Model**: `mask_detector.h5`  


## 3. Detection Layer
- **OpenCV Video Stream**: Captures frames from webcam  
- **Haar Cascades**: Detects face regions in each frame  
- **Integration**:
  - Extract face ROI  
  - Resize to CNN input size  
  - Run prediction using trained model  


## 4. Alert Layer
- **Logic**:
  - If prediction = “Mask”: Green bounding box + label  
  - If prediction = “No Mask”: Red bounding box + label + alert trigger  
- **Alert Options**:
  - On‑screen warning text  
  - Audio beep (via `winsound` or `playsound`)  
  - Logging non‑compliance events  


## 5. Deployment Layer (Optional)
- **Flask Web App**:
  - Serve detection results via browser  
  - REST API endpoint for predictions  
  - Lightweight UI for demo  


# Tech Stack
- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow / Keras  
- **Computer Vision Library**: OpenCV  
- **Face Detection**: Haar Cascades (XML classifier)  
- **Web Framework (Optional)**: Flask  
- **Data Handling**: NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  
- **Model Evaluation**: Scikit-learn (accuracy, confusion matrix, etc.)  
- **Alert System**: winsound / playsound (audio alerts)  



## 🔗 Workflow Diagram (Textual)


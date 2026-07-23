# What to Build
- A **Face Mask Detection System** that uses a webcam to identify whether individuals are wearing masks in real-time.
- A **Convolutional Neural Network (CNN)** trained on masked/unmasked face datasets.
- Integration with **OpenCV** and **Haar Cascades** for face detection.
- An **Alert System** that triggers warnings (visual/audio/log) when no mask is detected.
- Optional **Flask Deployment** for web-based demonstration.


# Targeted User
- **Educational Institutions**: Monitor compliance in classrooms or labs.
- **Corporate Offices**: Ensure workplace safety during health mandates.
- **Public Spaces**: Adaptable for malls, transport hubs, or events.
- **Developers & Researchers**: A learning project in computer vision and deep learning.



# Key Features
1. **Real-Time Detection**
   - Live webcam feed integration using OpenCV.
   - Haar Cascade face detection + CNN classification.

2. **Mask vs. No Mask Classification**
   - Trained CNN model (`mask_detector.h5`) predicts compliance.
   - High accuracy with preprocessed Kaggle dataset.

3. **Alert System**
   - Green bounding box + label for “Mask”.
   - Red bounding box + label + alert (sound/log) for “No Mask”.

4. **Lightweight Deployment**
   - Optional Flask app for browser-based demo.
   - REST API endpoint for predictions.

5. **Extensible Design**
   - Easy to retrain with new datasets.
   - Can integrate with IoT devices or security systems.


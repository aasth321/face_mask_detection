import numpy as np
import cv2
import tensorflow as tf
import os
import urllib.request

# 1. Configuration Constants
MODEL_PATH = "mask_detector.h5"  
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

if not os.path.exists(CASCADE_FILENAME):
    print("Downloading official face detection file...")
    url = f"https://githubusercontent.com{CASCADE_FILENAME}"
    urllib.request.urlretrieve(url, CASCADE_FILENAME)

face_cascade = cv2.CascadeClassifier(CASCADE_FILENAME)

# 2. Securely Load Your Custom Model
try:
    mask_model = tf.keras.models.load_model(MODEL_PATH)
    print("Model successfully loaded!")
except Exception as e:
    raise IOError(f"Could not open model file: {e}")

# Keras 3 Stable Input Dimensional Extraction
input_shape = mask_model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]

IMG_HEIGHT = int(input_shape[1])
IMG_WIDTH = int(input_shape[2])
CHANNELS = int(input_shape[3])

cap = cv2.VideoCapture(0)
print("\n==================================================")
print("Production inference active! Processing dual-neuron softmax layers.")
print("==================================================\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Pad bounding box to maintain proper aspect ratio limits
        pad = int(w * 0.1)
        y1, y2 = max(0, y - pad), min(h_frame, y + h + pad)
        x1, x2 = max(0, x - pad), min(w_frame, x + w + pad)
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            continue

        resized_face = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        if CHANNELS == 1:
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            normalized_face = gray_face / 255.0
            reshaped_face = np.expand_dims(normalized_face, axis=(0, -1))
        else:
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            normalized_face = rgb_face / 255.0
            reshaped_face = np.expand_dims(normalized_face, axis=0)

        # --- THE CORE ARCHITECTURE FIX ---
        # Obtain prediction arrays shaped exactly like [[prob_class_0, prob_class_1]]
        prediction = mask_model.predict(reshaped_face, verbose=0)
        pred_flat = prediction[0] # Strips batch dimension safely
        
        # Hardcoded classification index extraction matching your (None, 2) Dense array layout
        prob_class_0 = float(pred_flat[0])
        prob_class_1 = float(pred_flat[1])
        
        # Determine the maximum probability match using numpy native tools
        predicted_class_idx = np.argmax(pred_flat)
        confidence = float(pred_flat[predicted_class_idx])
        
        print(f"Neural Output -> [Class 0: {prob_class_0:.2%}] [Class 1: {prob_class_1:.2%}] -> Win: Class {predicted_class_idx}")

        # Map active logic flags based on the win index
        # Change '== 1' to '== 0' if your labels are still inverted in real-world testing!
        is_mask = (predicted_class_idx == 0)

        # --- UI DISPLAY RENDERING OVERLAYS ---
        if is_mask:
            label = f"Mask Detected ({confidence:.1%})"
            color = (0, 255, 0) # Clear Green
        else:
            label = f"No Mask Warning! ({confidence:.1%})"
            color = (0, 0, 255) # Sharp Red

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Live Stream Mask Identifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
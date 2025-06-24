# scripts/predict_pattern.py

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

IMG_SIZE = (160, 160)  # Match the training size

def predict_image_pattern(image_path, model_path='models/pattern_transfer_model.h5', data_dir='data/pattern_images/'):
    # Step 1: Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return

    # Step 2: Load and verify image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ OpenCV failed to load image: {image_path}")
        return

    # Step 3: Resize and normalize
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Step 4: Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return

    model = load_model(model_path)

    # Step 5: Predict
    preds = model.predict(img)
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    predicted_class = class_dirs[np.argmax(preds)]
    confidence = np.max(preds)

    print(f"🧠 Predicted Pattern: {predicted_class} ({confidence:.2f} confidence)")

if __name__ == "__main__":
    # Auto pick any image inside the folder
    folder = 'data/pattern_images/No_Pattern'
    files = [f for f in os.listdir(folder) if f.endswith('.png')]
    
    if not files:
        print("❌ No image files found in folder.")
    else:
        first_image = os.path.join(folder, files[0])
        predict_image_pattern(first_image)


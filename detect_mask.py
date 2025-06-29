import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def predict_mask(image_path: str, model_path: str) -> None:
    # Load model
    model = load_model(model_path)

    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image at {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    print(f"🧠 Faces found: {len(faces)}")

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)
        pred = model.predict(face_input)[0][0]

        label = "MASK ✅" if pred > 0.5 else "NO MASK ❌"
        confidence = round(pred if pred > 0.5 else 1 - pred, 3)
        color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{label} ({confidence})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Prediction Result")
    plt.show()

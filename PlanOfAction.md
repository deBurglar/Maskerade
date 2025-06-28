

# 🚀 Project: Face Mask Detection Web App

**Goal:** Real-time webcam-based app to detect whether a person is wearing a face mask or not.
**Tech:** TensorFlow, OpenCV, Streamlit
**Mode:** Jupyter Notebook (for learning) + Python Scripts (for real-time app)

---

## 📁 Step-by-Step Roadmap (with workspace guidance)

---

### 🔶 **PHASE 1: Project Setup**

#### ✅ Step 1: 📦 Install Required Libraries

🛠️ **Where:** Terminal / Anaconda Prompt

```bash
pip install tensorflow opencv-python streamlit numpy matplotlib
```

---

#### ✅ Step 2: 🗂️ Create Folder Structure

🛠️ **Where:** File Explorer or VS Code

```
face-mask-detector/
├── notebooks/
│   └── explore_model.ipynb
├── model/
│   └── mask_detector.model     ← (We’ll add this later)
├── app.py                      ← Streamlit app
├── detect_mask.py              ← Core prediction logic
└── utils.py                    ← Optional helper functions
```

---

### 🔶 **PHASE 2: Understand the Model**

#### ✅ Step 3: 📓 Load & Test Pretrained Model

🛠️ **Where:** `notebooks/explore_model.ipynb`

1. Load a sample image
2. Resize to model input shape (e.g. 224×224)
3. Normalize and predict

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("../model/mask_detector.model")

img = cv2.imread("test.jpg")  # add a sample test image
img_resized = cv2.resize(img, (224, 224)) / 255.0
img_batch = np.expand_dims(img_resized, axis=0)
prediction = model.predict(img_batch)
print("Prediction:", prediction)
```

✅ Use `matplotlib` to visualize the prediction.

---

#### ✅ Step 4: 🧠 Add Face Detection on Static Images

🛠️ **Where:** `explore_model.ipynb`

* Use OpenCV’s Haar cascades to detect faces

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 4)
```

* Loop over each face, crop it, and send it to model for prediction.

---

### 🔶 **PHASE 3: Build Prediction Logic**

#### ✅ Step 5: 🧱 Modularize into Python Function

🛠️ **Where:** `detect_mask.py`

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("model/mask_detector.model")

def predict_mask(face_image):
    resized = cv2.resize(face_image, (224, 224)) / 255.0
    batch = np.expand_dims(resized, axis=0)
    result = model.predict(batch)[0][0]
    return "Mask" if result > 0.5 else "No Mask"
```

✅ Test this by importing into Jupyter or calling directly from a test script.

---

### 🔶 **PHASE 4: Build the Streamlit App**

#### ✅ Step 6: 🖥️ Create the Real-Time App

🛠️ **Where:** `app.py`

```python
import streamlit as st
import cv2
from detect_mask import predict_mask

st.title("🧼 Face Mask Detector (Live)")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        label = predict_mask(face)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    FRAME_WINDOW.image(frame, channels="BGR")
```

✅ Tip: Use `Streamlit` widgets like checkboxes, buttons to control flow.

---

### 🔶 **PHASE 5: Test & Polish the App**

#### ✅ Step 7: 🧪 Run the app locally

🛠️ **Where:** Terminal

```bash
streamlit run app.py
```

✅ Ensure:

* Webcam starts/stops properly
* Correct predictions are displayed
* App layout is neat

---

### 🔶 **PHASE 6: Deployment (Optional for Resume)**

#### ✅ Step 8: 🚀 Deploy to HuggingFace Spaces or Streamlit Cloud

🛠️ **Where:** Your GitHub + HuggingFace/Streamlit account

* Push code to GitHub
* Connect repo to [Streamlit Cloud](https://streamlit.io/cloud) or [HuggingFace Spaces](https://huggingface.co/spaces)

---

### 🔶 **PHASE 7: Document & Showcase**

#### ✅ Step 9: 📄 Create README.md

🛠️ **Where:** `README.md` in your project root

Include:

* ✅ What the app does
* 🛠️ Tech stack
* 🧠 Model source
* 📸 Screenshots / demo video
* 🚀 Live deployment link

---

## ✅ Summary Table: Where to Do What

| Phase          | File/Tool              | Purpose                    |
| -------------- | ---------------------- | -------------------------- |
| Setup          | Terminal + Folder      | Set up project             |
| Explore model  | `explore_model.ipynb`  | Learn how model works      |
| Face detection | `explore_model.ipynb`  | Apply on static images     |
| Predict logic  | `detect_mask.py`       | Write reusable functions   |
| UI/App         | `app.py`               | Build real-time webcam app |
| Run App        | `streamlit run app.py` | Launch interface           |
| Deployment     | GitHub + HuggingFace   | Make it public             |
| Showcase       | README.md              | For resume & portfolio     |

---


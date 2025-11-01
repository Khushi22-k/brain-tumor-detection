# 🧠 Brain Tumor Detection using Deep Learning

## 📘 Project Overview
This project focuses on **automated brain tumor classification** from MRI images using **Deep Learning**.  
The goal is to assist radiologists and healthcare professionals in **early and accurate detection** of brain tumors through an AI-powered model.

The system classifies MRI images into four categories:
- 🧩 **Pituitary Tumor**
- 🧩 **Glioma**
- 🧩 **Meningioma**
- 🧩 **No Tumor**

---

## 🚀 Key Features
✅ Deep learning model built with **Convolutional Neural Networks (CNN)**  
✅ Implemented **Transfer Learning** using **VGG16** for superior accuracy  
✅ Preprocessed and augmented MRI images to improve model generalization  
✅ Real-time prediction interface for image uploads  
✅ Evaluation metrics include **Accuracy, Precision, Recall, and F1-Score**

---

## 🧑‍💻 Tech Stack
- **Language:** Python  
- **Frameworks & Libraries:** TensorFlow, Keras, NumPy, OpenCV, Matplotlib  
- **Model Architecture:** CNN (VGG16-based transfer learning)  
- **Dataset:** Brain MRI Dataset (Kaggle / Custom medical dataset)

---

## 📊 Model Workflow

1. **Data Preprocessing**
   - Image resizing to 128x128
   - Normalization and augmentation
   - Train-test split (80-20)

2. **Model Building**
   - Base model: VGG16 (pretrained on ImageNet)
   - Added fully connected dense layers
   - Softmax output for 4-class classification

3. **Training**
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy
   - Metrics: Accuracy

4. **Evaluation**
   - Tested on unseen MRI images
   - Visualized confusion matrix and classification report

---

## 🧪 Results

| Metric | Value |
|---------|--------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~95% |
| Loss | Decreased steadily over epochs |

🖼️ **Sample Prediction Results**
---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/brain-tumor-detector.git
cd brain-tumor-detector
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Model
bash
Copy code
python brain_tumor_detector.py
4️⃣ Predict on Custom Image
python
Copy code
from keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('brain_tumor_model.h5')
img = load_img('path_to_image.jpg', target_size=(128,128))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
prediction = model.predict(x)
print(prediction)
🧠 Future Enhancements
Integration with a Flask web interface for live image uploads

Deploy model using Streamlit / FastAPI

Add Grad-CAM visualization for model interpretability

Experiment with ResNet50 and EfficientNet architectures

📂 Project Structure
bash
Copy code
brain-tumor-detector/
│
├── dataset/
│   ├── train/
│   ├── test/
│
├── brain_tumor_detector.ipynb
├── README.md
└── model.h5
🩺 Acknowledgments
Dataset sourced from Kaggle: Brain MRI Dataset

Inspired by ongoing research in AI for Healthcare

🧾 License
This project is licensed under the MIT License — free to use and modify for educational and research purposes.

🤝 Connect with Me
👩‍💻 Khushi Sharma
 Linkedin:-https://www.linkedin.com/in/khushi-sharma-2b4897289
Email:- khushi.sharma9119@gmail.com



import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load the trained model (from the notebook)
model = load_model('model.h5')

# Define class labels matching the label_map from the notebook
class_labels = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor']

# Streamlit app
st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (matching the notebook's prediction logic)
    img = load_img(uploaded_file, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize as in training augmentation
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence_score = np.max(predictions)

    # Determine class
    predicted_label = class_labels[predicted_class_index]
    if predicted_label == 'no_tumor':
        result = "No Tumor Detected"
    else:
        result = f"Tumor Detected: {predicted_label}"

    # Display the result (adapted from notebook's detect_and_display)
    st.write(f"{result} (Confidence: {confidence_score * 100:.2f}%)")

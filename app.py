import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

st.set_page_config(page_title="Debug Mode", layout="wide")
st.title("🛠️ AI Diagnostic Mode")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    classes = joblib.load('plant_disease_classes.pkl')
    return model, classes

model, class_names = load_model()

uploaded_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_to_scan = Image.open(uploaded_file).convert('RGB')
    st.image(image_to_scan, caption="Uploaded Image", width=300)
    
    # Base Image Array
    img = image_to_scan.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- The 3 Mathematical Transformations ---
    # 1. Raw Pixels (0 to 255)
    img_raw = img_array.copy()
    
    # 2. Rescaled (0 to 1)
    img_255 = img_array.copy() / 255.0
    
    # 3. MobileNetV2 Standard (-1 to 1)
    img_mobile = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
    
    # --- Get Predictions ---
    pred_raw = model.predict(img_raw)
    pred_255 = model.predict(img_255)
    pred_mobile = model.predict(img_mobile)
    
    # --- UI Layout ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Test 1: Raw (0 to 255)")
        idx = np.argmax(pred_raw)
        st.write(f"**Guess:** {class_names[idx]}")
        st.write(f"**Confidence:** {np.max(pred_raw)*100:.2f}%")
        
    with col2:
        st.subheader("Test 2: Rescaled (0 to 1)")
        idx = np.argmax(pred_255)
        st.write(f"**Guess:** {class_names[idx]}")
        st.write(f"**Confidence:** {np.max(pred_255)*100:.2f}%")
        
    with col3:
        st.subheader("Test 3: MobileNetV2 (-1 to 1)")
        idx = np.argmax(pred_mobile)
        st.write(f"**Guess:** {class_names[idx]}")
        st.write(f"**Confidence:** {np.max(pred_mobile)*100:.2f}%")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

st.set_page_config(page_title="Plant Disease Scanner", page_icon="🌿", layout="centered")

st.title("🌿 AI Plant Disease Scanner")
st.write("Upload a photo or take a picture of a plant leaf to detect potential diseases.")
st.info("""
**💡 How to get the best results:**
This AI was trained in a controlled lab environment. For the highest accuracy:
1. Pluck the leaf from the plant.
2. Place it flat on a **plain white or solid-colored background** (like a piece of paper).
3. Ensure good lighting and center the leaf in the frame.
""")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    classes = joblib.load('plant_disease_classes.pkl')
    return model, classes

try:
    model, class_names = load_model()
except Exception as e:
    st.error("Model files not found. Please ensure .h5 and .pkl files are uploaded.")
    st.stop()

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take a Picture"])
image_to_scan = None

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_scan = Image.open(uploaded_file)

with tab2:
    camera_photo = st.camera_input("Take a picture of the leaf")
    if camera_photo is not None:
        image_to_scan = Image.open(camera_photo)

if image_to_scan is not None:
    st.markdown("---")
    st.image(image_to_scan, caption="Image to Scan", use_container_width=True)
    st.write("🔄 Analyzing leaf structure and health...")
    
    image_to_scan = image_to_scan.convert('RGB')
    img = image_to_scan.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100
    
    clean_label = predicted_class.replace("___", " - ").replace("_", " ")
    
    st.subheader("📋 Scan Results")
    
    if confidence < 80.0:
        st.warning("⚠️ **Diagnosis Unclear**")
        st.write(f"The model is only **{confidence:.2f}%** confident in its analysis, which is too low to provide a reliable diagnosis.")
        st.write("🔍 **Reason:** The background might be too cluttered, or the leaf is not centered. Please try again using a plain background.")

    else:
        if "healthy" in clean_label.lower():
            st.success(f"**Diagnosis:** {clean_label}")
            st.balloons()
        else:
            st.error(f"**Diagnosis:** {clean_label}")
            
        st.write(f"**Confidence:** {confidence:.2f}%")

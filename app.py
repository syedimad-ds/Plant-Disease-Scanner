import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(page_title="Plant Disease Scanner", page_icon="🌿")
st.title("🌿 AI Plant Disease Scanner")
st.write("Upload a photo or take a picture of a plant leaf to detect potential diseases.")

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    return model

try:
    model = load_model()
except Exception as e:
    st.error("Model file not found. Please ensure plant_disease_model.h5 is uploaded.")
    st.stop()

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take a Picture"])
image_to_scan = None

with tab1:
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_scan = Image.open(uploaded_file)

with tab2:
    camera_photo = st.camera_input("Take a picture of the leaf")
    if camera_photo is not None:
        image_to_scan = Image.open(camera_photo)

if image_to_scan is not None:
    st.markdown("---")
    st.image(image_to_scan, caption="Image to Scan", use_container_width=True)
    st.write("🔄 Scanning...")
    
    img = image_to_scan.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = img_array / 255.0
   
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions) * 100
    
    clean_label = predicted_class.replace("___", " - ").replace("_", " ")
    
    st.subheader("📋 Scan Results")
    if "healthy" in clean_label.lower():
        st.success(f"**Diagnosis:** {clean_label}")
        st.balloons()
    else:
        st.error(f"**Diagnosis:** {clean_label}")
        
    st.info(f"**Confidence:** {confidence:.2f}%")


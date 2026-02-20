# 🌿 AI Plant Disease Scanner

## 📌 Project Overview
This project is an interactive web application that uses Deep Learning and Computer Vision to diagnose plant diseases from photos of leaves. By automating the detection process, this tool aims to help farmers, agronomists, and everyday gardeners identify crop illnesses early, reducing yield loss and pesticide overuse. 

## ⚙️ Tech Stack
* **Deep Learning:** TensorFlow / Keras (MobileNetV2)
* **Computer Vision:** Pillow (PIL), NumPy
* **Web Deployment:** Streamlit
* **Model Serialization:** HDF5 (.h5) & Joblib (.pkl)

## 🚀 Live Demo
Check out the live web application here: **[https://plant-disease-scanner-k2.streamlit.app/]**

# 📊 Dataset
Kaggle Dataset: **[https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset]**

## 🧠 How It Works
The brain of this application is a Convolutional Neural Network (CNN) built using Transfer Learning.
1. **Transfer Learning:** I utilized Google's `MobileNetV2`, a highly efficient, pre-trained image classification model. I froze the base layers and attached a custom Dense classification head to identify specific plant diseases.
2. **Data Pipeline:** The model was trained using a high-speed `tf.data` pipeline with dynamic prefetching to optimize GPU usage. Image augmentation (random flips, rotations) was applied to improve model generalization.
3. **Inference UI:** Users can either drag-and-drop an image or use their device's camera. The app resizes the image to 224x224 pixels, converts it to a NumPy array, and passes it through the model to output a diagnosis and confidence percentage in real-time.

## 📂 Repository Structure
* `app.py`: The Streamlit web application script.
* `plant_disease_model.h5`: The saved CNN model.
* `plant_disease_classes.pkl`: The exact category labels mapped to the model's output neurons.
* `requirements.txt`: The dependencies required to run the app.

## 💻 How to Run Locally
1. Clone this repository to your local machine.
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## ⚠️ Model Limitations & Real-World Applicability
While this model achieves high validation accuracy on the test set, deploying computer vision to physical agricultural environments presents unique challenges:

* **Background Noise & Segmentation:** The training dataset consists primarily of isolated leaves on plain, contrasting backgrounds (like a white sheet of paper). If a user takes a photo of a leaf attached to a tree with complex backgrounds (dirt, other plants, shadows), the model's confidence may drop significantly. 
* **Multiple Concurrent Diseases:** The network uses a `softmax` activation function, which forces the model to choose exactly *one* condition. In reality, a single plant can simultaneously suffer from a fungal infection, pest damage, and a nutrient deficiency.
* **Lighting Conditions:** Harsh shadows, glare from the sun, or poor smartphone camera quality can alter the pixel arrays enough to cause misclassification.

**Future Improvements:** To make this production-ready for commercial farming, the architecture should be upgraded from an Image Classifier to an Object Detection model (like YOLOv8) or an Image Segmentation model (like Mask R-CNN). This would allow the AI to draw bounding boxes exactly around the diseased spots on a leaf, ignoring background noise entirely.

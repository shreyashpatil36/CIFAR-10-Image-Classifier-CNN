import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the pretrained Keras .h5 model
@st.cache_resource
def load_cifar10_model():
    model = load_model('cifar10_model.h5')
    return model

model = load_cifar10_model()

def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((32, 32))            # CIFAR-10 size
    img = img.convert('RGB')              # Ensures 3 channels
    img_array = np.array(img) / 255.0     # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image (jpg, jpeg, png). The app predicts one of the CIFAR-10 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    st.write("Classifying...")
    img_array = preprocess_image(uploaded_file)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(tf.nn.softmax(predictions[0]))
    st.write(f"**Prediction:** {predicted_class} ({confidence*100:.2f}% confidence)")
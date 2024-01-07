import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
from tensorflow.keras.models import load_model
import seaborn as sns

# Load the pre-trained model
model = load_model('plant_species_model.h5')  # Update with the correct path

# Constants
img_size = 224
flowers = ['banana', 'coconut', 'corn', 'mango', 'orange', 'paddy', 'papaya', 'pineapple', 'sweet potatoes', 'watermelon']

# Streamlit App
st.title("Plant Species Prediction")

# Upload Image through Streamlit
uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Read and preprocess the image
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_arr = cv2.resize(img, (img_size, img_size))
    st.image(resized_arr, caption="Uploaded Image.", use_column_width=True)

    # Model Prediction
    x = preprocess_input(np.array([resized_arr]))
    x = x.reshape(-1, img_size, img_size, 3)
    pred = model.predict(x)
    prediction_label = flowers[np.argmax(pred)]

    # Display prediction result
    st.subheader("Prediction:")
    if prediction_label == 'banana':
        st.write("The predicted plant species is: banana")
    elif prediction_label == 'coconut':
        st.write("The predicted plant species is: coconut")
    elif prediction_label == 'corn':
        st.write("The predicted plant species is: corn")
    elif prediction_label == 'mango':
        st.write("The predicted plant species is: mango")
    elif prediction_label == 'orange':
        st.write("The predicted plant species is: orange")
    elif prediction_label == 'paddy':
        st.write("The predicted plant species is: paddy")    
    elif prediction_label == 'papaya':
        st.write("The predicted plant species is: papaya")
    elif prediction_label == 'pineapple':
        st.write("The predicted plant species is: pineapple")
    elif prediction_label == 'sweet potatoes':
        st.write("The predicted plant species is: sweet potatoes")
    elif prediction_label == 'watermelon':
        st.write("The predicted plant species is: watermelon")
    else:
        pass
    # Display prediction probabilities
    pred_results = pd.DataFrame(data=pred, columns=flowers)
    st.subheader("Prediction Probabilities:")
    st.bar_chart(pred_results.T)

# Note: Adjust the model loading part based on the actual method used to load your model.

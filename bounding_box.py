#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('cnn_bounding_box_model.h5')

# Define input size the model expects
input_size = (224, 224)

def preprocess_image(image):
    """
    Preprocesses the image before passing it to the model.
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_resized = cv2.resize(img, input_size)  # Resize to match model input
    img_normalized = img_resized.astype(np.float32) / 255.0  # Normalize the image
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    return img, img_input

def predict_bounding_box(image):
    """
    Predicts the bounding box for the image using the trained model.
    """
    # Get original image and preprocessed image
    original_img, img_input = preprocess_image(image)
    # Predict the bounding box
    predicted_bbox = model.predict(img_input)[0]
    return original_img, predicted_bbox

def visualize_prediction(image):
    """
    Visualizes the predicted bounding box on the image.
    """
    # Get original image and predicted bounding box
    original_img, predicted_bbox = predict_bounding_box(image)
    height, width, _ = original_img.shape
    x, y, w, h = predicted_bbox * [width, height, width, height]  # Denormalize bbox

    # Draw bounding box on the image
    img_with_bbox = original_img.copy()
    cv2.rectangle(img_with_bbox, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Display the image with bounding box
    plt.imshow(img_with_bbox)
    plt.axis('off')
    plt.title('Predicted Bounding Box')
    plt.show()

# Streamlit app
def main():
    st.title('Object Detection - Bounding Box Prediction')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Make prediction and show the result
        if st.button('Predict Bounding Box'):
            st.text("Predicting the bounding box...")

            # Call the visualization function
            visualize_prediction(image)
        else:
            st.text("Upload an image and click 'Predict Bounding Box' to see the result.")

if __name__ == '__main__':
    main()


# In[ ]:





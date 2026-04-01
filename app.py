import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image

# Load models

logistic_regression_path = os.path.join('models', 'logistic_regression.pkl')
svm_path = os.path.join('models', 'svm.pkl')
kmeans_path = os.path.join('models', 'kmeans.pkl')
with open(logistic_regression_path, "rb") as f:
    logistic_model = pickle.load(f)
with open(svm_path, "rb") as f:
    svm_model = pickle.load(f)

with open(kmeans_path, "rb") as f:
    kmeans = pickle.load(f)


# ---- SIFT Feature Extraction ----
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors


# ---- Create Bag of Words Histogram ----
def create_bow_histogram(descriptors, kmeans):
    
    k = kmeans.n_clusters
    histogram = np.zeros(k)

    if descriptors is not None:
        clusters = kmeans.predict(descriptors)

        for c in clusters:
            histogram[c] += 1

    return histogram.reshape(1, -1)


# ---- Streamlit UI ----
st.title("Animal Image Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Extract SIFT
    descriptors = extract_sift_features(gray)
    descriptors = descriptors.astype(float)

    # Convert to BoW
    bow_features = create_bow_histogram(descriptors, kmeans)

    # Prediction
    logistic_prediction = logistic_model.predict(bow_features)
    svm_prediction = svm_model.predict(bow_features)
    st.subheader("Prediction")
    st.write(logistic_prediction[0])
    st.write(svm_prediction[0])
    

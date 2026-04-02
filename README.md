# 🐱🐶 Cat vs Dog Image Classification App

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end computer vision application that classifies images of cats and dogs using **classical computer vision techniques** (SIFT + Bag of Visual Words) instead of deep learning. Deployed with an interactive web interface using Streamlit.

## 🎯 Live Demo

> *Coming soon: Deployed on Streamlit Cloud*

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Workflow](#example-workflow)
- [Real-World Applications](#real-world-applications)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

## 🔍 Project Overview

This project demonstrates a full machine learning pipeline that solves image classification using **traditional computer vision methods**. Unlike modern deep learning approaches (CNNs), this implementation relies on:

- **SIFT** (Scale-Invariant Feature Transform) for robust feature extraction
- **Bag of Visual Words** (BoVW) for image representation
- **Classical ML classifiers** (SVM/Random Forest) for prediction

The model achieves competitive accuracy without requiring GPUs, making it lightweight and interpretable.


### Pipeline Steps:

1. **Image Preprocessing**  
   Convert to grayscale, resize, and normalize

2. **Feature Extraction**  
   Detect keypoints and compute SIFT descriptors for each image

3. **Vocabulary Building**  
   Cluster all SIFT descriptors from training images using KMeans to create a "visual vocabulary" (e.g., 500 visual words)

4. **Image Encoding**  
   Convert each image into a histogram counting how many times each visual word appears

5. **Classification**  
   Train a machine learning model (SVM/Random Forest) on the histogram representations

6. **Prediction**  
   New images go through the same pipeline for real-time classification

## ✨ Features

- 🖼️ **Interactive Web Interface** - Upload images via Streamlit
- 📊 **Real-time Predictions** - Instant cat/dog classification
- 🔧 **Classical CV Pipeline** - No deep learning dependencies
- 💾 **Persistent Model** - Trained model and KMeans saved as `.pkl` files
- 📈 **Modular Code** - Well-organized components for easy modification
- 🚀 **Lightweight** - Runs on CPU without GPU requirements

## 📁 Project Structure
Cat-Dog-Classification/
│
├── app.py # Streamlit web application
├── model.pkl # Trained classifier
├── kmeans.pkl # KMeans vocabulary model
├── requirements.txt # Dependencies
│
├── data/
│ ├── train/ # Training images (cat/, dog/ subfolders)
│ │ ├── cat/
│ │ └── dog/
│ └── test/ # Test images
│
├── src/
│ ├── components/
│ │ ├── data_ingestion.py # Load and organize dataset
│ │ ├── feature_extraction.py # SIFT + BoVW pipeline
│ │ └── model_training.py # Train and save classifier
│ │
│ └── utils/ # Helper functions
│
└── README.md


## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cat-dog-classification.git
cd cat-dog-classification

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate

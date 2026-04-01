## I have to have multiples 

import os 
import sys
import cv2
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



@dataclass
class ModelTrainerConfig():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path: str = os.path.join(root_dir, 'models')


class ModelTrainer():
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
        

    def feature_creation(self,files):
        try:
            descriptions = []
            for file in files:
                desc = np.loadtxt(file)
                if desc.ndim ==1:
                    desc = desc.reshape(1, -1)
                
                descriptions.append(desc)

            data = np.vstack(descriptions)
            
            os.makedirs(self.model_train_config.model_path, exist_ok=True)

            kmeans = KMeans(n_clusters = 50, random_state=47, n_init=10)
            kmeans.fit(data)


            kmeans_path = os.path.join(self.model_train_config.model_path, "kmeans.pkl")
            with open(kmeans_path, "wb") as f:
                pickle.dump(kmeans, f)


            feature_hist = []
            for image_desc in descriptions:
                clusters = kmeans.predict(image_desc) 
                hist, _ = np.histogram(clusters, bins=np.arange(51))
                hist = hist.astype(float)
                hist /= (hist.sum() + 1e-6)
                feature_hist.append(hist)

            feature_hist = np.array(feature_hist)

            return feature_hist
        except Exception as e:
            raise CustomException(e, sys)

    def modeltrain(self, X_train_features, y_train):
        try:
            print("Model training")
            y_train_correct = [0 if y == 'dog' else 1 for y in y_train]
            
            models = {
                "logistic_regression": LogisticRegression(),
                "svm": SVC(kernel="linear")
            }


            for name, model in models.items():
                model.fit(X_train_features, y_train_correct)

                model_path = os.path.join(self.model_train_config.model_path, f"{name}.pkl")

                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                print(f"{name} saved to {model_path}")
        except Exception as e:
            raise CustomException(e, sys)
    
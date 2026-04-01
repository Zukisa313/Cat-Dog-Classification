import os 
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
## access the files in features test 

### convert them to feature vectors using the existing kmeans them implement prediction

@dataclass
class PredictConfig:
    root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    test_features_path: str = os.path.join(root_dir, 'data','features', "test")
    models_path: str = os.path.join(root_dir,'models')

class Predict:
    def __init__(self):
        self.predict_config = PredictConfig()
    

    def predictTestData(self):
        try:
            kmeans_path = os.path.join(self.predict_config.models_path, "kmeans.pkl")
            with open(kmeans_path,'rb') as f:
                kmeans_model = pickle.load(f)


            logistic_path = os.path.join(self.predict_config.models_path, "logistic_regression.pkl")
            with open(logistic_path,'rb') as f:
                logistic_model = pickle.load(f)

            svm_path = os.path.join(self.predict_config.models_path, 'svm.pkl')
            with open(svm_path, 'rb') as f:
                sv_model = pickle.load(f)
            classes = ["cat", "dog"]
            X_test = []
            y_test = []

            for cls in classes:
                class_path = os.path.join(self.predict_config.test_features_path, cls)
                files = [f for f in os.listdir(class_path) if f.endswith(".txt")]

                for file in files:
                    file_path = os.path.join(class_path, file)
                    desc = np.loadtxt(file_path)
                    if desc.ndim ==1:
                        desc = desc.reshape(1, -1)
                    
                    X_test.append(desc)
                    y_test.append(0 if "dog" in cls else 1)
            
            X_test_hist = []
            for images_desc in X_test:
                clusters = kmeans_model.predict(images_desc)
                hist, _ = np.histogram(clusters, bins = np.arange(51))
                hist = hist.astype(float)
                hist /= (hist.sum()+ 1e-6)
                X_test_hist.append(hist)
            
            X_test_hist = np.array(X_test_hist)
            y_test= np.array(y_test)

            print(f"Features shapes {X_test_hist.shape}")
            print(f"Labels shape: {y_test.shape}")

            y_pred_svm = sv_model.predict(X_test_hist)
            y_pred_logistic = logistic_model.predict(X_test_hist)

            return (y_pred_svm, y_pred_logistic, y_test)



        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    predictions = Predict()
    y_pred_svm, y_pred_logistic , y_test= predictions.predictTestData()
    print("="*25)
    print("Logistic Regression results")
    score_log = accuracy_score(y_test, y_pred_logistic)
    confusion_log = confusion_matrix(y_test, y_pred_logistic)
    print(f"Accuracy score {score_log}")
    print(f"Confusion matric {confusion_log}")
    print("-"*25)
    print("SVC results")
    score_svm = accuracy_score(y_test, y_pred_svm)
    confusion_svm = confusion_matrix(y_test, y_pred_svm)
    print(f"Accuracy score {score_svm}")
    print(f"Confusion matric {confusion_svm}")
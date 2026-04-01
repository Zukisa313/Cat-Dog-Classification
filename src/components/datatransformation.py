import os 
import sys
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    desc_data_path: str = os.path.join(root_dir, 'data','feature')

class Datatransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def initiate_data_transformation(self,X_train, X_test):
        try:
            sift = cv2.SIFT_create(nfeatures=1000)
            X_train_features = []
            X_test_features = []
            y_test_features = []
            y_train_features = []
            print("Here")
            print(len(X_train))
            for path in X_train:
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                _, descriptors = sift.detectAndCompute(gray,None)
                img_path = Path(path)

                filename = img_path.stem
                label = img_path.parent.name
                save_path =  Path(
                    r'C:\\Users\\Zukisa\\Documents\\Cat Dog Classification\\data\\features\\train'
                        ) / label / f"{filename}.txt"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                X_train_features.append(save_path)
                y_train_features.append(label)
                np.savetxt(save_path, descriptors)
            logging.info("X_train sift detection complete")

            for path in X_test:
                img = cv2.imread(path)
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                _, descriptors = sift.detectAndCompute(gray,None)
                img_path = Path(path)

                filename = img_path.stem
                label = img_path.parent.name
                save_path =  Path(
                    r'C:\\Users\\Zukisa\\Documents\\Cat Dog Classification\\data\\features\\test'
                        ) / label / f"{filename}.txt"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                X_test_features.append(save_path)
                y_test_features.append(label)
                np.savetxt(save_path, descriptors)
            
            logging.info("X_test sift detection complete")
            
            return (
                X_train_features, X_test_features, y_train_features, y_test_features
            )
                    
                
        except Exception as e:
            raise CustomException(e, sys)

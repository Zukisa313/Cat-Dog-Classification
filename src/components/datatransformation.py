import os 
import sys
import cv2
import numpy as np

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    desc_data_path: str = os.path.join(root_dir, 'data','feature')
    raw_data_path: str = os.path.join(root_dir, 'data','animals')

class Datatransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def initiate_data_transformation(self):
        try:
            logging.info("Initiating data transformation")
            ## read data 
            input_dir = self.data_transformation_config.raw_data_path
            output_dir = self.data_transformation_config.desc_data_path

            sift =cv2.SIFT_create(nfeatures = 1000)
            for label in ["cat", "dog"]:
                input_path = os.path.join(input_dir,label)
                output_path = os.path.join(output_dir,label)
                os.makedirs(output_path, exist_ok=True)
                logging.info("Access image files")
            for file in os.listdir(input_path):
                img_path = os.path.join(input_path, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Extract features
                keypoints, descriptors = sift.detectAndCompute(gray, None)

                if descriptors is None:
                    print(f"No features found in {file}")
                    continue

                filename = os.path.splitext(file)[0] + ".txt"
                save_path = os.path.join(output_path, filename)
                np.savetxt(save_path, descriptors)
                
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_split(self):
        pass 
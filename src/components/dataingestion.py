import os 
import sys
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_data_path: str = os.path.join(root_dir, 'data','animals')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def data_split(self):
        print("About to execute")
        datasetpath = self.data_ingestion_config.raw_data_path
        paths = []
        labels = []
        for label in os.listdir(datasetpath):
            class_folder = os.path.join(datasetpath,label)

            for img in os.listdir(class_folder):
                paths.append(os.path.join(class_folder,img))
                labels.append(label)

        X_train, X_test, y_train, y_test = train_test_split(
            paths, labels, test_size=0.2, stratify=labels, random_state=42
        )

        return ( X_train, X_test, y_train, y_test

        )


if __name__ == "__main__":
    data = DataIngestion()
    X_train, X_test , y_train, y_test = data.data_split()
    print(y_train)

    


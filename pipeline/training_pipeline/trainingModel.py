"""
This is the Entry point for Training the Machine Learning Model.
Written By: Rachit More
Version: 1.0
Revisions: None
"""
# Doing the necessary imports
import sys
import datetime
import warnings
from Pipeline.app_tracking.logger import App_Logger
from Pipeline.app_tracking.exception import AppException
from Pipeline.Utils.utils import FileOperation
from .src.Stage1_data_ingestion.data_loader import Data_Getter
from .src.Stage2_DataValidation.DataTypeValidation import RawDataValidation,PrePreocessedDataValidation
from .src.Stage3_DataPreprocessing.DataTransformation import DataPreprocessing
from .src.Stage4_model_building.model_building import ModelTraining
from .src.Stage5_model_evaluation.model_evaluation import ModelEvaluation

# Suppress the specific warning
warnings.filterwarnings("ignore" )

def income_model_on_local():
    print("Code flow start")
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    Data_Getter(current_time).ingest_from_csv()
    RawDataValidation(current_time)
    DataPreprocessing(current_time)
    PrePreocessedDataValidation(current_time)
    ModelTraining(current_time)
    ModelEvaluation(current_time)

def income_model_on_database():
    print("Code flow start")
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    Data_Getter(current_time).ingest_from_database()
    RawDataValidation(current_time)
    DataPreprocessing(current_time)
    PrePreocessedDataValidation(current_time)
    ModelTraining(current_time)
    ModelEvaluation(current_time)
                
if __name__ == "__main__":
    try:
        income_model_on_database()
    except:
        income_model_on_local()

    


        
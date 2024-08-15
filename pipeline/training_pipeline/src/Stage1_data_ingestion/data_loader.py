import pandas as pd
import sys
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException
from utils.utils import FileOperation

class Data_Getter:
    """
    This class shall be used for obtaining the data from the source for training.
    Written By: Rachit More
    Version: 1.0
    Revisions: None
    """
    def __init__(self, current_time):
        try:
            print("Data Ingestion has been Started")
            self.current_time = current_time
            self.filepath = f"artifacts/logs/Stage1_Dataingestion/{self.current_time}.txt"
            self.logging = App_Logger(self.filepath)
            self.fileOperation = FileOperation()
            self.fileOperation.delete_files_in_directory("artifacts/data")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e

    # Data ingestion from specific path in csv file
    def ingest_from_csv(self,path = "artifacts/data/raw/adult.csv"):
        try:
            self.logging.log("Data ingestion started from specific path in csv")
            df = pd.read_csv(path)
            self.fileOperation.save_data_to_csv(df,"artifacts/data/raw.csv")
            self.logging.log("Data ingested successfully from specific path in csv")

        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e

        
    
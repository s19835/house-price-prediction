import zipfile
from abc import ABC, abstractmethod
import pandas as pd

# Here we are using factory design pattern

# 1. Base class (interface)
class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a file"""
        pass

# file types --> .csv, .json, .xlsx

# 2. Concreate class
class CSVProcessor(DataProcessor):
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Extract data and return file as dataframe"""
        return pd.read_csv(file_path)
    
class JSONProcessor(DataProcessor):
    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_json(file_path)
    
class XLSXProcessor(DataProcessor):
    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_excel(file_path)

class DataProcessorFactory:
    @staticmethod
    def get_processor(file_extension):
        if file_extension == '.csv':
            return CSVProcessor()
        elif file_extension == '.json':
            return JSONProcessor()
        elif file_extension == '.xlsx':
            return XLSXProcessor()
        else :
            raise ValueError("Unsupported file type")



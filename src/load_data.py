import zipfile
from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

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

# 3. Factory Class
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


# Define data loading
def load_file(file_path: str) -> pd.DataFrame:
    '''Function for load the data file into the df'''
    try:
        file_extension = file_path[file_path.rfind('.'):]
        logger.info(f"loading file {file_path} (Extension: {file_extension})")

        processor = DataProcessorFactory.get_processor(file_extension)
        data = processor.load_data(file_path)

        logger.info(f"Successfully loaded data. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise




#example = load_file("./data/raw/test.csv")
# example = load_file("./data/raw/csvjson.json")
# print(example)
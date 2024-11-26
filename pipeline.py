from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from bson.json_util import dumps
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import logging
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("model_pipeline.log"),  
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataMongoDB:

    """
    A class to handle MongoDB connection and data extraction for fraud detection.
    
    This class establishes connection with MongoDB and provides functionality
    to extract and transform data into pandas DataFrame.
    """
    def __init__(self):
        """
        Initialize MongoDB connection to localhost and select the appropriate database.
        Raises exception if connection fails.
        """
        logger.info("Establishing Connection with MongoDB and extracting the data for the model.")
        try:
            self.client = MongoClient("mongodb://localhost:27017/")
            self.db = self.client["local"]
            self.collections = self.db["inueron.ai data"]
            logger.info("Initial Connection established, now extracting data from Mongo DB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
    
    def data_extraction(self):
        '''
        Extracts the data and converts into a pandas dataframe for efficient data preparation, feature engineering and model building.
        '''
        try:
            self.records = [dumps(data) for data in self.collections.find()]
            self.dataframe = pd.read_json(f"[{','.join(self.records)}]")
            logger.info("Succesfully converted into a df")
        except Exception as e:
            logger.error(f"Failed to convert into a pandas dataframe: {e}")
        
class MachineLearningModel(DataMongoDB):
    def __init__(self):
        super().__init__()
        
    def data_cleaning(self):
        '''
        Performs Data Cleaning, Data Manipulation, Feature Creation for appropriate data format for the model.
        '''
        self.data_extraction()
        logger.info("Data Manipulation and Extraction.")
        self.dataframe['TX_DATETIME'] = self.dataframe['TX_DATETIME'].apply(lambda x: datetime.fromisoformat(x['$date'].replace('Z', '+00:00')))
        self.dataframe['TX_FRAUD'] = self.dataframe['TX_FRAUD'].apply(lambda x: 0 if x == "Legitimate" else 1)
        self.dataframe['TX_MONTH'] = pd.DatetimeIndex(self.dataframe['TX_DATETIME']).month
        if '_id' in self.dataframe.columns:
            self.dataframe.drop(columns=['_id'], inplace=True)
        self.mean_amount = self.dataframe['TX_AMOUNT'].mean()
        self.dataframe['TX_AMOUNT'].fillna(self.mean_amount, inplace=True)
        self.mean_seconds = self.dataframe['TX_TIME_SECONDS'].mean()
        self.dataframe['TX_TIME_SECONDS'].fillna(self.mean_seconds, inplace=True)
        self.mean_days = self.dataframe['TX_TIME_DAYS'].mean()
        self.dataframe['TX_TIME_DAYS'].fillna(self.mean_days, inplace=True)
        def is_night(timestamp): 
            tx_hour = timestamp.hour 
            is_night = tx_hour <= 6 
            return int(is_night)
        def is_weekend(timestamp):
            tx_weekend = timestamp.weekday()
            is_weekend = tx_weekend >= 5
            return int(is_weekend)
        self.dataframe['TX_NIGHT'] = self.dataframe['TX_DATETIME'].apply(is_night)
        self.dataframe['TX_WEEKEND'] = self.dataframe['TX_DATETIME'].apply(is_weekend)
        self.dataframe['TX_IS_AMOUNT_HIGH'] = self.dataframe['TX_AMOUNT'].apply(lambda x: 1 if x >= 150 else 0)
        self.dataframe['TX_TIME_TAKEN_HIGH'] = self.dataframe['TX_TIME_SECONDS'].apply(lambda x: 1 if x > 7903233.708571933 else 0)
        self.dataframe['TX_HOUR'] = self.dataframe['TX_DATETIME'].dt.hour
        self.dataframe['TX_RUSH_HOUR'] = self.dataframe['TX_HOUR'].apply(lambda x: 1 if x in [8,9,10,16,17,18] else 0)
        self.dataframe['TX_AMOUNT_ROUNDED'] = self.dataframe['TX_AMOUNT'].apply(lambda x: 1 if x % 100 == 0 else 0)
        self.dataframe['TX_LUNCH_TIME'] = self.dataframe['TX_HOUR'].apply(lambda x: 1 if x in [11,12,13,14] else 0)
        self.dataframe['TX_LATE_NIGHT'] = self.dataframe['TX_HOUR'].apply(lambda x: 1 if x in [23,0,1,2,3] else 0)
        fraud_scenario_dummies = pd.get_dummies(self.dataframe['TX_FRAUD_SCENARIO'], prefix="FRAUD_SCENARIO", dtype = int)
        self.dataframe = pd.concat([self.dataframe, fraud_scenario_dummies], axis=1)
        logger.info("Data cleaning,extraction and Manipulation Completed.")
        
    def outlier_detection(self):
        '''
        Peforms the appropriate outlier detection and caps them using capping method. Here the data is very important and cannot be trimmed hence capped.
        '''
        self.data_cleaning()
        logger.info("Start Outlier Treatment")
        self.mean = self.dataframe['TX_AMOUNT'].mean()
        self.std = self.dataframe['TX_AMOUNT'].std()
        self.threshold = 3
        self.outliers = []
        self.twentyfithpercentile = np.percentile(self.dataframe['TX_AMOUNT'], 25)
        self.seventyfifthpercentile = np.percentile(self.dataframe['TX_AMOUNT'], 75)
        self.outlier_count = 0
        for index,data in self.dataframe['TX_AMOUNT'].items():
            self.z_score = (data - self.mean) / self.std
            if self.z_score > self.threshold:
                self.dataframe.at[index, 'TX_AMOUNT'] = self.seventyfifthpercentile
                self.outlier_count += 1
            elif self.z_score < -self.threshold:
                self.dataframe.at[index, 'TX_AMOUNT'] = self.twentyfithpercentile
                self.outlier_count += 1
        if self.outlier_count == 0:
            logger.info("No Outliers Detected")
        logger.debug(f"No of Outliers treated: {self.outlier_count}")
        logger.info("Outlier Treatment Completed, utilised capping method.")
        
    def train_and_test(self):
        '''
        Split the data for training and testing.
        '''
        self.outlier_detection()
        logger.info("Initialising the data for the Gradient Boosting Trees.")
        self.X = self.dataframe.drop(columns = ['TX_FRAUD', 'TX_FRAUD_SCENARIO', 'TX_FRAUD', 'TX_DATETIME', 'TX_FRAUD_SCENARIO', 'FRAUD_SCENARIO_Large Amount', 'FRAUD_SCENARIO_Leaked data', 'FRAUD_SCENARIO_Legitimate', 'FRAUD_SCENARIO_Random Fraud'])
        self.y = self.dataframe['TX_FRAUD']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size = 0.4, random_state = 42, stratify = self.y)
        logger.info("Completed Dataset Preparation for training and testing.")
        
    def training(self):
        '''
        Performs training on the Gradient Boosting Trees for optimal machine learning performance.
        '''
        self.train_and_test()
        logger.info("Training Started")
        smote = SMOTE(random_state=42)
        X_train_smo, y_train_smo = smote.fit_resample(self.X_train, self.y_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smo)
        X_test_scaled = scaler.transform(self.X_test)
        logger.info("Preparing the Gradient Boosting Trees for model prediction.")
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train_scaled, y_train_smo)
        self.y_pred = self.model.predict(X_test_scaled)
        logger.info("Training Completed")
        print("Classification Report:", classification_report(self.y_test, self.y_pred))
        
    def best_model_selected(self):
        '''
        Pickling the best model.
        '''
        logger.info("Looking for the best model to pickle, in case to be used for front end.")
        self.training()
        try:
            with open("best_model.pkl", "wb") as model_file:
                pickle.dump(self.model, model_file) 
                logger.debug("Best model saved as 'best_model.pkl', incase the model is to be used with a front end.")
        except Exception as e:
            logger.error(f"Failed to Pickle the best model: {e}")
        
        
MLPipeline = MachineLearningModel()
MLPipeline.best_model_selected()
import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.exception import Custom_Exception
from src.logger import logging
from data_ingestion import DataIngestion
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object
class DataTransformationConfig:
    preprocessr_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transforation_config = DataTransformationConfig()
    
    def get_tranformer_object(self):
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_educatoin","lunch","test_preparation_course"]
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "median")),
                    ("scaler",StandardScaler()),

                ]
            )
            cat_pipeline = Pipeline(
                    steps = [
                    ("imputer",SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encodeer",OneHotEncoder()),
                    ("scaler",StandardScaler())                        

                    ]

                )
            logging.info("Numerical columns standard Scalling")
            logging.info("categorical columns encoding info")

            preprocessor = ColumnTransformer(
                    [
                        ("num_pipeline",num_pipeline,numerical_columns),
                        ("cat_pipelines",cat_pipeline,categorical_columns)
                    ]

                )
            return preprocessor
        except Exception as e :
            raise Custom_Exception(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")
            logging.info("opening preprocessing objects")

            preprocessing_object = self.get_tranformer_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis = 1)
            target_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)

            logging.info("Applying preprocessing objecy on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_train_df)


            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_test_df)]     
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transforation_config.preprocessr_obj_file_path,  
                obj = preprocessing_object

            )
            return (
                train_arr,
                test_arr,
                self.data_transforation_config.preprocessr_obj_file_path    

            )
        except Exception as e:

            raise Custom_Exception(e,sys)

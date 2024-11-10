import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from src.exception import Custom_Exception
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

class DataTransformationConfig:
    preprocessr_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transforation_config = DataTransformationConfig()
    
    def get_tranformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            
            # Numerical pipeline with standard scaling
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            
            # Categorical pipeline with one-hot encoding and scaling (with_mean=False)
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(sparse=False)),  # Set sparse=False
                ("scaler", StandardScaler(with_mean=False))  # Set with_mean=False for compatibility
            ])
            
            logging.info("Numerical columns standard scaling")
            logging.info("Categorical columns encoding info")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            
            return preprocessor
        except Exception as e:
            raise Custom_Exception(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")
            logging.info("Initializing preprocessing objects")

            preprocessing_object = self.get_tranformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Drop the target column to separate input features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[[target_column_name]]  # Corrected syntax

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[[target_column_name]]  # Corrected syntax

            logging.info("Applying preprocessing object on training and testing dataframes")

            # Transform input features
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            # Combine input features and target into arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transforation_config.preprocessr_obj_file_path,  
                obj=preprocessing_object
            )
            return train_arr, test_arr
        except Exception as e:
            raise Custom_Exception(e, sys)

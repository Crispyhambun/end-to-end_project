import os
import sys
from dataclasses import dataclass
from src.utils import eval_model
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_name = os.path.join("artifacts", "model.pkl")

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting the data into train and test sets")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],  # Fixing test_arr indexing here
                test_arr[:, -1]
            )
            
            # Defining models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),  # Changed to AdaBoostRegressor
            }

            # Evaluating models
            model_report = eval_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
            best_model_score = max(model_report.values())

            # Identifying the best model by name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Custom_Exception("No satisfactory model found with an R^2 score above 0.6")
            
            logging.info("Best model found on both training and testing dataset")

            # Saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_name,
                obj=best_model
            )

            # Making predictions with the best model
            predictions = best_model.predict(x_test)
            r2_square = r2_score(y_test, predictions)
            logging.info(f"Best Model: {best_model_name} with R^2 score: {r2_square}")
            
            return r2_square
        
        except Exception as e:
            raise Custom_Exception(e, sys)

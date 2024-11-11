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
from sklearn.model_selection import GridSearchCV
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
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            # Defining models and their hyperparameters
            models = {
                "Random Forest": {
                    "model": RandomForestRegressor(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10]
                    }
                },
                "Decision Tree": {
                    "model": DecisionTreeRegressor(),
                    "params": {
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10]
                    }
                },
                "Gradient Boosting": {
                    "model": GradientBoostingRegressor(),
                    "params": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7]
                    }
                },
                "Linear Regression": {
                    "model": LinearRegression(),
                    "params": {}
                },
                "K-Neighbors Regressor": {
                    "model": KNeighborsRegressor(),
                    "params": {
                        "n_neighbors": [3, 5, 10],
                        "weights": ['uniform', 'distance']
                    }
                },
                "XGB Regressor": {
                    "model": XGBRegressor(),
                    "params": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5, 7]
                    }
                },
                "CatBoost Regressor": {
                    "model": CatBoostRegressor(verbose=False),
                    "params": {
                        "iterations": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "depth": [6, 8, 10]
                    }
                },
                "AdaBoost Regressor": {
                    "model": AdaBoostRegressor(),
                    "params": {
                        "n_estimators": [50, 100],
                        "learning_rate": [1.0, 0.5, 0.1]
                    }
                },
            }

            # Evaluating models with hyperparameter tuning
            best_model = None
            best_model_score = -1

            for model_name, model_info in models.items():
                model = model_info["model"]
                params = model_info["params"]
                
                if params:
                    grid_search = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
                    grid_search.fit(x_train, y_train)
                    best_model_score_temp = grid_search.best_score_
                    best_model_temp = grid_search.best_estimator_
                else:
                    model.fit(x_train, y_train)
                    best_model_score_temp = model.score(x_test, y_test)
                    best_model_temp = model
                
                logging.info(f"{model_name} Best Score: {best_model_score_temp}")
                print(f"{model_name} Best Score: {best_model_score_temp}")
                if best_model_score_temp > best_model_score:
                    best_model_score = best_model_score_temp
                    best_model = best_model_temp

            if best_model_score < 0.6:
                raise Custom_Exception("No satisfactory model found with an R^2 score above 0.6")
            
            logging.info("Best model found on both training and testing dataset")
            print("Best model found on both training and testing dataset is")
            # Saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_name,
                obj=best_model
            )

                      # Making predictions with the best model
            predictions = best_model.predict(x_test)
            
            # Calculating the R² score for the predictions
            r2_square = r2_score(y_test, predictions)
            
            # Logging the best model and its R² score
            print(f"Best Model: {best_model.__class__.__name__} with R^2 score: {r2_square}")
            # Returning the R² score
            return r2_square
        
        except Exception as e:
            raise Custom_Exception(e, sys)
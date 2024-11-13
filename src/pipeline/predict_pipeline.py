import sys
from src.exception import Custom_Exception
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self) -> None:
        pass
    def predict(self,feature):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled =preprocessor.transform(feature)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise Custom_Exception(e,sys)


class Custom_Data:
    def __init__(self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education:str,
        test_preparation_course:str,
        lunch:str,
        reading_score:int,
        writing_score:int):

        self.gender = gender 
        self.race_ethnicity= race_ethnicity
        self.parental_level_of_education= parental_level_of_education
        self.test_preparation_course= test_preparation_course
        self.reading_score= reading_score
        self.writing_score= writing_score
        self.lunch = lunch

    def get_data_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender' :[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score],
                'lunch':[self.lunch]
                   }
            
            return  pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise Custom_Exception(e,sys) 
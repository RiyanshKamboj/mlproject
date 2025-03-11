import os 
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model_name=list(models.keys())[i]
            print("Model Name: ",model_name)
            logging.info(f"Model Name: {model_name}")
            para=param[list(models.keys())[i]]
           
            gs=GridSearchCV(model,para, cv=5)
            gs.fit(X_train,y_train)
            print(f"Best Params for{model_name}: ",gs.best_params_)
            logging.info(f"Best params for {model_name} are {gs.best_params_}")
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)

            y_test_pred=model.predict(X_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            print("Train model R2_score: ",train_model_score,"  ", "Test Model R2_score",test_model_score)
            logging.info(f"Train model R2_score: {train_model_score} Test Model R2_score {test_model_score}")

            report[list(models.keys())[i]]=test_model_score
        
        return report







    except Exception as e:
        raise CustomException(e,sys)

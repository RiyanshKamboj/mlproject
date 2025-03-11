import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from src.logger import logging
from src.exception import CustomException



from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Splitting Train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models={
                "SVM_Regressor_without_params":SVR(),
                "SVM Regressor":SVR(),
                "Linear Regression": LinearRegression(),
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGBoostRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "Adaboost Regressor": AdaBoostRegressor()
            }
            params={
                "SVM_Regressor_without_params":{},
                "SVM Regressor":{
                    "kernel": ["linear", "rbf"],  # Type of kernel function
                   # "C": [0.1, 1, 10, 100],  # Regularization parameter
                    "epsilon": [0.01, 0.1, 0.2, 0.5],  # Epsilon in the epsilon-SVR model
                    "gamma": ["scale", "auto", 0.01, 0.1, 1],  # Kernel coefficient (only for ‘rbf’, ‘poly’, and ‘sigmoid’)
                    "degree": [2,3],  # Degree of polynomial kernel (only for ‘poly’ kernel)
                    "coef0": [0, 0.1, 0.5, 1],  # Independent term in kernel function (used in ‘poly’ and ‘sigmoid’ kernels)
                    "shrinking": [True, False],  # Whether to use shrinking heuristic
                    "tol": [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criterion
                    #"max_iter": [-1, 1000]  # Hard limit on iterations (-1 means no limit)

                },
                "Logistic Regression" :{
                    "penalty": [ "elasticnet", None],  # Different regularization techniques
                    #"C": [0.01, 0.1, 10, 100],  # Regularization strength
                    "solver": ["liblinear", "lbfgs", "saga"],  # Different solvers
                    "max_iter": [100, 200, 500, 1000],  # Iteration limits
                    #"class_weight": [None, "balanced"],  # Handling class imbalance
                    #"multi_class": ["auto", "ovr", "multinomial"],  # Multi-class strategies
                    #"l1_ratio": [None, 0.1, 0.5, 0.9]  # ElasticNet mixing parameter (only for 'elasticnet' penalty)
                    },

                "Linear Regression":{
    
                    "fit_intercept": [True, False],  # Whether to calculate the intercept or not
                    #"normalize": [True, False],  # Deprecated in newer versions; use StandardScaler instead
                    "copy_X": [True, False],  # Whether to copy X or modify it in place
                    #"n_jobs": [-1, 1, None],  # Number of CPU cores to use (-1 means all cores)
                    "positive":[True,False]               
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    "n_estimators": [100, 200, 500],  # Number of boosting stages
                  # "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
                    "max_depth": [3, 6, 10],  # Maximum depth of individual trees
                   # "min_samples_split": [2, 5, 10],  # Minimum samples required to split a node
                   # "min_samples_leaf": [1, 3, 5],  # Minimum samples required per leaf node
                   # "subsample": [0.6, 0.8, 1.0],  # Fraction of samples used per tree
                    #"max_features": ["auto", "sqrt", "log2", None],  # Features considered for split
                   # "loss": ["squared_error", "absolute_error", "huber"],  # Loss function
                  #  "criterion": ["friedman_mse", "squared_error"],  # Splitting criterion
                  #  "alpha": [0.75, 0.85, 0.95],  # Huber loss and quantile loss parameter
                   # "random_state": [42]  # Ensure reproducibility
                    },
                
                "K-Neighbours Regressor":{
                    "n_neighbors": [3, 5, 7, 10, 15],  # Number of neighbors to consider
                    "weights": ["uniform", "distance"],  # Weight function used in prediction
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # Algorithm to compute nearest neighbors
                    "leaf_size": [10, 20, 30, 40, 50],  # Leaf size for BallTree/KDTree
                    "p": [1, 2],  # Power parameter for Minkowski distance (1=Manhattan, 2=Euclidean)
                    #"metric": ["euclidean", "manhattan", "minkowski", "chebyshev"]  # Distance metric

                },
                "XGBoostRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    "booster": ["gbtree", "dart"],  # Type of boosting
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report: dict=evaluate_model(X_train=X_train,y_train=y_train,
                                              X_test=X_test,y_test=y_test,
                                              models=models, param=params)
            sorted_model_report=sorted(model_report.values())
            
            best_model_score=max(sorted_model_report)

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("Sorry, No best model found")
            logging.info(f"Best found model on both training and testing dataset{best_model_name} with R2_score: {best_model_score}")
            print(f"Best found model on both training and testing dataset{best_model_name} with R2_score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score


        except Exception as e:
            raise CustomException(e,sys)

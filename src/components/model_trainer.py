import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, 
                              GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artificats","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    def initiatie_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "Catboosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square

            ## Hyperparameter tuning of each model using pipeline
            pipeline = Pipeline([('regressor', RandomForestRegressor())])
            rf_model = RandomForestRegressor(random_state=42)
            param_distributions = [
                # Random Forest Regressor
                {
                    'regressor': [RandomForestRegressor()],
                    'regressor__n_estimators': randint(50, 200),
                    'regressor__max_depth': randint(3, 20)
                },
                # Decision Tree Regressor
                {
                    'regressor': [DecisionTreeRegressor()],
                    'regressor__max_depth': randint(3, 20),
                    'regressor__min_samples_split': randint(2, 20)
                },
                # Gradient Boosting Regressor
                {
                    'regressor': [GradientBoostingRegressor()],
                    'regressor__n_estimators': randint(50, 200),
                    'regressor__learning_rate': uniform(0.01, 0.2),
                    'regressor__max_depth': randint(3, 10)
                },
                # Linear Regression (no hyperparameters to tune)
                {
                    'regressor': [LinearRegression()]
                },
                # K-Neighbors Regressor
                {
                    'regressor': [KNeighborsRegressor()],
                    'regressor__n_neighbors': randint(1, 15),
                    'regressor__weights': ['uniform', 'distance']
                },
                # XGBoost Regressor
                {
                    'regressor': [XGBRegressor(objective='reg:squarederror')],
                    'regressor__n_estimators': randint(50, 200),
                    'regressor__learning_rate': uniform(0.01, 0.2),
                    'regressor__max_depth': randint(3, 10)
                },
                # CatBoost Regressor
                {
                    'regressor': [CatBoostRegressor(verbose=False)],
                    'regressor__iterations': randint(50, 200),
                    'regressor__learning_rate': uniform(0.01, 0.2),
                    'regressor__depth': randint(4, 10)
                },
                # AdaBoost Regressor
                {
                    'regressor': [AdaBoostRegressor()],
                    'regressor__n_estimators': randint(50, 200),
                    'regressor__learning_rate': uniform(0.01, 0.2)
                }
            ]

            # 4. Instantiate and fit RandomizedSearchCV
            # n_iter = 100 means it will run 100 random combinations in total across ALL models
            # 100 iterations will be distributed amongst the models in the param_distributions list
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions,
                n_iter=100,  
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1, # Use all available CPU cores
                random_state=42
            )
            random_search.fit(x_train, y_train)


        except Exception as e:
            raise CustomException(e,sys)
            
from dataclasses import dataclass
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor
)


SKlearnModels =Union[
    LinearRegression, Lasso, ElasticNet, KNeighborsRegressor, DecisionTreeRegressor, SVR, MLPRegressor, 
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor,
]


@dataclass
class ParamsKfold:
    models: List[SKlearnModels]
    X_train: pd.DataFrame
    Y_train: pd.Series
    X_test: pd.DataFrame
    Y_test: pd.Series
    num_folds: int
    scoring: int
    seed: int = 0
    shuffle: bool = False


@dataclass
class ResultKfold:
    names: List[str]
    kfold_results: List[np.ndarray]
    train_results: List[np.float64]
    test_results: List[np.float64]


def run_kfold_analysis(params: ParamsKfold) -> ResultKfold:
    names = []
    kfold_results = []
    test_results = []
    train_results = []
    for name, model in params.models:
        names.append(name)
        
        ## K-Fold analysis:    
        kfold = (
            KFold(n_splits=params.num_folds, shuffle=True, random_state=params.seed)
            if params.shuffle else
            KFold(n_splits=params.num_folds)
        )
        
        # Converted mean square error to positive. The lower the beter
        cv_results = (-1) * cross_val_score(model, params.X_train, params.Y_train, cv=kfold, scoring=params.scoring)
        kfold_results.append(cv_results)
        
        # Full Training period
        res = model.fit(params.X_train, params.Y_train)
        train_result = mean_squared_error(res.predict(params.X_train), params.Y_train)
        train_results.append(train_result)
        
        # Test results
        test_result = mean_squared_error(res.predict(params.X_test), params.Y_test)
        test_results.append(test_result)
        
        print(f"{name}: {cv_results.mean():.6f} ({cv_results.std():.6f}) {train_result:.6f} {test_result:.6f}")

    return ResultKfold(names, kfold_results, train_results, test_results)

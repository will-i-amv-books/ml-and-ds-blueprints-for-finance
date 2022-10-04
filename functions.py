from dataclasses import dataclass
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
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


@dataclass
class ParamsGridSearch:
    model: SKlearnModels
    params_grid: Dict[str, List]
    X_train: pd.DataFrame
    Y_train: pd.Series
    X_test: pd.DataFrame
    Y_test: pd.Series
    num_folds: int
    scoring: int
    seed: int = 0
    shuffle: bool = False


def run_grid_search(params: ParamsGridSearch) -> GridSearchCV:
    kfold = (
        KFold(n_splits=params.num_folds, shuffle=True, random_state=params.seed)
        if params.shuffle else
        KFold(n_splits=params.num_folds)
    )
    grid = GridSearchCV(
        estimator=params.model, 
        param_grid=params.params_grid, 
        scoring=params.scoring, 
        cv=kfold
    )
    grid_result = grid.fit(params.X_train, params.Y_train)
    print(f"Best: {grid_result.best_score_:.6f} using {grid_result.best_params_}")

    for mean, stdev, param in zip(
        grid_result.cv_results_['mean_test_score'], 
        grid_result.cv_results_['std_test_score'], 
        grid_result.cv_results_['params']
    ):
        print(f"{mean:.6f} ({stdev:.6f}) with: {param}")

    return grid_result


def calc_best_features(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    best_features = SelectKBest(k=5, score_func=f_regression)
    for col in Y.columns:
        fit = best_features.fit(X, Y[col])
        df_scores = pd.DataFrame(fit.scores_)
        df_columns = pd.DataFrame(X.columns)

        # Concat two dataframes for better visualization
        feature_scores = pd.concat([df_columns, df_scores], axis=1)
        feature_scores.columns = ['Specs', 'Score']
        print('----------------------------------------------------------------')
        print(feature_scores.nlargest(10, 'Score').set_index('Specs'))

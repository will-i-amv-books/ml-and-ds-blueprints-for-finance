import itertools as it
from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np
import pandas_datareader.data as web
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class LstmParams:
    X_train: pd.DataFrame
    neurons: int
    learn_rate: float
    momentum: int


@dataclass
class ArimaParams:
    X_train: pd.DataFrame
    Y_train: pd.DataFrame
    order: List[int] = field(default_factory=list)


def load_data(start, end, params):
    data = {}
    for label, param in params.items():
        data[label] = web.DataReader(
            name=param['tickers'],
            data_source=param['data_source'],
            start=start,
            end=end,
        )

    return data


def clean_data(data, period):
    periods = [period, period * 3, period * 6, period * 12]
    x1 = (
        np
        .log(data['stk'].loc[:, ('Adj Close', ('GOOGL', 'IBM'))])
        .diff(period)
        .droplevel(0, axis=1)
    )
    x2 = np.log(data['ccy']).diff(period)
    x3 = np.log(data['idx']).diff(period)
    x4 = (
        pd
        .concat(
            [
                np.log(data['stk'].loc[:, ('Adj Close', 'MSFT')]).diff(period)
                for period in periods
            ],
            axis=1
        )
        .dropna()
    )
    x4.columns = ['MSFT_DT', 'MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT']
    x = pd.concat([x1, x2, x3, x4], axis=1)

    y = (
        data['stk']
        .loc[:, ('Adj Close', 'MSFT')]
        .apply(np.log)
        .diff(period)
        .shift(-period)
        .to_frame()
        .droplevel(0, axis=1)
    )
    y.columns = y.columns + '_pred'

    dataset = pd.concat([y, x], axis=1).dropna().iloc[::period, :]
    X = dataset.loc[:, x.columns]
    Y = dataset.loc[:, y.columns]

    return X, Y


def evaluate_arima_model(params: ArimaParams) -> np.float64:
    """
    Evaluate an ARIMA model for a given order (p, d, q).
    """
    modelARIMA = ARIMA(
        endog=params.Y_train,
        exog=params.X_train,
        order=params.order
    )
    model_fit = modelARIMA.fit()
    return mean_squared_error(params.Y_train, model_fit.fittedvalues)


def run_grid_search_arima(params: ArimaParams) -> Tuple[int]:
    best_score = float("inf")
    best_cfg = None
    for p, d, q in it.product(range(0, 3), range(0, 2), range(0, 2)):
        params.order = (p, d, q)
        try:
            mse = evaluate_arima_model(params)
            if mse < best_score:
                best_score, best_cfg = mse, params.order
            print(f'ARIMA {params.order} MSE={mse:.7f}')
        except:
            continue

    print(f'Best ARIMA {best_cfg} MSE={best_score:.7f}')
    return best_cfg


def create_lstm_model(params: LstmParams) -> Sequential:
    # Create model
    model = Sequential()
    model.add(LSTM(
        units=50,
        input_shape=(params.X_train.shape[1], params.X_train.shape[2])
    ))

    # More number of cells can be added if needed
    model.add(Dense(1))
    optimizer = SGD(lr=params.learn_rate, momentum=params.momentum)
    model.compile(loss='mse', optimizer='adam')

    return model

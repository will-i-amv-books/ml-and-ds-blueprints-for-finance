from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class ParamsCallOPtion:
    true_alpha: float
    true_beta: float
    true_sigma0: float
    risk_free_rate: float


def calc_opt_volatility(
    moneyness: np.ndarray, 
    time_to_maturity: np.ndarray,
    params: ParamsCallOPtion
) -> np.ndarray:
    return (
        params.true_sigma0 + 
        (params.true_alpha * time_to_maturity) + 
        (params.true_beta * np.square(moneyness - 1))
    )


def calc_opt_price(
    moneyness: np.ndarray, 
    time_to_maturity: np.ndarray, 
    option_vol: np.ndarray,
    params: ParamsCallOPtion
) -> np.ndarray:
    x1 = np.log(1 / moneyness)
    x2 = params.risk_free_rate + np.square(option_vol)
    x3 = params.risk_free_rate - np.square(option_vol)
    x4 = option_vol * np.sqrt(time_to_maturity)

    d1 = (x1 + (x2 * time_to_maturity)) / x4
    d2 = (x1 + (x3 * time_to_maturity)) / x4
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    price = N_d1 - (moneyness * np.exp((-1) * params.risk_free_rate * time_to_maturity) * N_d2)
    return price

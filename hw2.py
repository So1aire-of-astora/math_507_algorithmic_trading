import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize

class DynamicStrategy:
    def __init__(self, price_data, **kwargs) -> None:
        self.returns = self.get_returns(price_data)
        self.M = self.returns.shape[0]
        self.num_days = 250
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_returns(self, price, log_return = False) -> pd.Series:
        rets = price.pct_change()[1:]
        if log_return:
            rets = np.log1p(rets)
        return rets
    
    def get_params(self, returns) -> np.array:
        mu = returns.mean() * self.num_days
        sigma = returns.std(ddof = 1) * np.sqrt(self.num_days)
        return np.array([mu, sigma])
    
    def get_optimal_alpha(self, mu, sigma):
        init = [1., 0.]
        obj_func = lambda x: .5*(np.power((1 + x@np.array([self.rf, mu+sigma])), self.zeta) + 
                                np.power((1 + x@np.array([self.rf, mu-sigma])), self.zeta))
        constr = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        opt = scipy.optimize.minimize(fun = obj_func, x0 = init, constraints = constr)
    
    



def main():
    dt = yf.download(tickers = "^GSPC", start = "2014-01-01", end = "2022-01-01")
    strategy = DynamicStrategy(price_data = dt["Adj Close"], rf = .01, zeta = -3, d = 2, N = 1000, T = 100)
    breakpoint()

if __name__ == "__main__":
    main()
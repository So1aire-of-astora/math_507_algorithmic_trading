import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize

class DynamicStrategy:
    def __init__(self, price_data, log_returns = False, **kwargs) -> None:
        self.log_returns = log_returns
        self.returns = self.get_returns(price_data)
        self.M = self.returns.shape[0]
        self.num_days = 250
        self.alphas = pd.DataFrame(columns = ["riskless", "risky"])
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_returns(self, price) -> pd.Series:
        rets = price.pct_change()[1:]
        if self.log_returns:
            rets = np.log1p(rets)
        return rets
    
    def get_params(self, returns) -> np.array:
        # if self.log_returns:
        #     mu = returns.mean() * self.num_days
        # else: 
        #     mu = (np.power((1 + returns).prod(), 1/returns.shape[0]) - 1) * self.num_days
        mu = returns.mean() * self.num_days
        sigma = returns.std(ddof = 1) * np.sqrt(self.num_days)
        return np.array([mu, sigma])
    
    def get_optimal_alpha(self, params) -> float:
        assert params.shape[0] == 2
        mu, sigma = params[0], params[1]
        init = [1., 0.]
        obj_func = lambda x: -.5 * (np.power((1 + x@np.array([self.rf, mu+sigma])), self.zeta) +
                                np.power((1 + x@np.array([self.rf, mu-sigma])), self.zeta))
        constr = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        opt = scipy.optimize.minimize(fun = obj_func, x0 = init, constraints = constr)
        return opt.x
    
    def backtest(self):
        for i in range(self.N, self.M, self.T):
            print(i)
            train_set = self.returns[i - self.N : i]
            test_idx = self.returns.index[i : i + self.T]
            train_params = self.get_params(train_set)
            alpha = self.get_optimal_alpha(train_params)
            self.alphas = pd.concat([self.alphas, pd.DataFrame(data = [alpha] * test_idx.shape[0], index = test_idx, columns = self.alphas.columns)])
            print(alpha)

        pnl = (1 + self.alphas["riskless"] * (np.power(1 + self.rf, 1/self.num_days) - 1) + self.alphas["risky"] * self.returns[self.N:]).cumprod() - 1
        mu_trade = ...
        sigma_trade = ...



def main():
    dt = yf.download(tickers = "^GSPC", start = "2014-01-01", end = "2022-01-01")
    strategy = DynamicStrategy(price_data = dt["Adj Close"], log_returns = False, rf = .01, zeta = -3, d = 2, N = 1000, T = 100)
    strategy.backtest()
    breakpoint()

if __name__ == "__main__":
    main()
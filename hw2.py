import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize
import scipy.interpolate
from functools import singledispatchmethod

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Strategy:
    def __init__(self, price_data, log_returns = False, **kwargs) -> None:
        self.log_returns = log_returns
        self.returns = self.get_returns(price_data)
        self.M = self.returns.shape[0]
        self.num_days = 250
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.r_eff = self.rf / self.num_days

    def get_returns(self, price) -> pd.Series:
        rets = price.pct_change()[1:]
        if self.log_returns:
            rets = np.log1p(rets)
        return rets
    
    def get_train_params(self, returns, annualize = False):
        mu = returns.mean() 
        sigma = returns.std(ddof = 1) 
        if annualize:
            mu *= self.num_days
            sigma *= np.sqrt(self.num_days)
        return mu, sigma
    
    def get_optimal_alpha(self, mu, sigma, window):
        raise NotImplementedError
    
    @singledispatchmethod
    def trade_eval(self, data):
        '''
        get params based on the wealth process
        '''
        raise NotImplementedError("Input type unsupported.")
    @trade_eval.register
    def _(self, data: np.ndarray):
        mu_trade, sigma_trade = self.get_train_params((data[1:] - data[:-1]) / data[:-1], annualize = True)
        return pd.DataFrame(data = [[mu_trade, sigma_trade, (mu_trade - self.rf) / sigma_trade]], columns = ["mu", "sigma", "Sharpe"])  
    @trade_eval.register
    def _(self, data: pd.Series):
        mu_trade, sigma_trade = self.get_train_params((data).pct_change().iloc[1:], annualize = True)
        return pd.DataFrame.from_dict({data.index[-1]: [mu_trade, sigma_trade, (mu_trade - self.rf) / sigma_trade]}, \
                                      orient = "index", columns = ["mu", "sigma", "Sharpe"]) 

    def tc(self, a, alpha_target, wealth):
        raise NotImplementedError
    
    def update_wealth(self, alpha, curr_wealth, curr_pos, returns):
        raise NotImplementedError

    def backtest(self, verbose = 1):
        trade_params = pd.DataFrame(columns = ["mu", "sigma", "Sharpe"])
        wealth = np.array([1.])
        curr_pos = 0
        for i in range(self.N, self.M, self.T):
            train_set = self.returns[i - self.N : i]
            test_set = self.returns[i + 1 : i + self.T + 1]
            mu, sigma = self.get_train_params(train_set)

            wealth_curr = wealth[-1]
            alpha_curr = self.get_optimal_alpha(mu, sigma, test_set.shape[0])
            wealth_arr, curr_pos = self.update_wealth(alpha_curr, wealth_curr, curr_pos, test_set)
            wealth = np.append(wealth, wealth_arr)
            curr_params = self.trade_eval(wealth_arr)
            trade_params = pd.concat([trade_params, curr_params], axis = 0)

        eval_overall = self.trade_eval(wealth)
        trade_params = pd.concat([trade_params, eval_overall.reset_index(drop = True).rename(index = {0: "overall"})], axis = 0)

        self.pnl = pd.Series(wealth - 1, index = self.returns.index[-wealth.shape[0]:])
        if verbose:
            print("Annualized mean, std dev and Sharpe Ratio: \n{}".format(trade_params))

    def plot_pnl(self):
        assert hasattr(self, "pnl")
        plt.plot(self.pnl)
        plt.title("PnL Process of Strategy")
        plt.xticks(rotation = 45)
        plt.xlabel("time")
        plt.ylabel("wealth")
        plt.show()
        
class DynamicStrategy(Strategy):
    def __init__(self, price_data, log_returns = False, **kwargs) -> None:
        super().__init__(price_data, log_returns, **kwargs)
    
    def get_optimal_alpha(self, mu, sigma, window = None):
        init = .5
        obj_func = lambda x: -.5 * (np.power(1 + (1 - x)*self.r_eff + x * (mu + sigma), self.zeta) 
                                    + np.power(1 + (1 - x)*self.r_eff + x * (mu - sigma), self.zeta))
        opt = scipy.optimize.minimize(fun = obj_func, x0 = init)
        return opt.x[0]

    def tc(self, a, alpha_target, wealth):
        return 0
    
    def update_wealth(self, alpha, curr_wealth, curr_pos, returns): 
        T = returns.shape[0]
        wealth_arr = np.zeros(T)
        for t in range(T):
            a = curr_pos * (1 + returns[t]) / (1 + (1 - curr_pos) * self.r_eff + curr_pos * returns[t])
            curr_wealth = curr_wealth * (1 + (1 - alpha) * self.r_eff + alpha * returns[t]) - self.tc(a, alpha, curr_wealth)
            wealth_arr[t] = curr_wealth
            curr_pos = alpha
        return wealth_arr, curr_pos

class DynamicStrategyTC(DynamicStrategy):
    def __init__(self, price_data, log_returns=False, **kwargs):
        super().__init__(price_data, log_returns, **kwargs)
    
    def tc(self, a, alpha_target, wealth):
        assert hasattr(self, "lambda_")
        return self.lambda_ * wealth * np.abs(a - alpha_target)

class DynamicStrategyGrid(Strategy):
    def __init__(self, price_data, log_returns=False, **kwargs) -> None:
        super().__init__(price_data, log_returns, **kwargs)

    def set_grid(self):
        assert hasattr(self, "grid")
        return np.linspace(*self.grid), np.linspace(*self.grid)
    
    def curr_rwd(self, alphas: np.ndarray, a: np.ndarray, mu: float, epsilon: float) -> np.ndarray:
        return np.power((1 + alphas * (mu+epsilon) - self.lambda_ * abs(alphas[None,:] - a[:, None])), self.zeta)

    def get_next_state(self, alpha: np.ndarray, mu, epsilon) -> np.ndarray:
        return (alpha * (1+mu+epsilon)) / (1 + (1-alpha) * self.r_eff + alpha * (mu+epsilon))

    def rand_argmax(self, arr):
        return np.random.choice(np.where(arr == arr.max())[0])

    def get_optimal_alpha(self, mu, sigma, window):
        a_grid, alpha_grid = self.set_grid()
        V_grid = np.zeros([self.grid[-1], window + 1])
        strategy_grid = np.zeros([self.grid[-1], window])
        V_grid[:, -1] = 1 / self.zeta

        for t in reversed(range(window)):
            interp_func = scipy.interpolate.interp1d(a_grid, V_grid[:, t + 1], kind = "linear", fill_value = "extrapolate")
            '''
            dimension: [a, alpha, prob]
            '''
            rwd = np.stack([self.curr_rwd(alpha_grid, a_grid, mu, sigma), self.curr_rwd(alpha_grid, a_grid, mu, -sigma)], axis = 2)
            a_next = np.vstack([self.get_next_state(alpha_grid, mu, sigma), self.get_next_state(alpha_grid, mu, -sigma)])
            V_exp = (rwd * interp_func(a_next).T[None,:,:]).mean(axis = 2)
            V_grid[:, t] = V_exp.max(axis = 1)
            strategy_grid[:, t] = np.apply_along_axis(func1d = self.rand_argmax, arr = V_exp, axis = 1)
            
        return alpha_grid[strategy_grid.astype(int)]

    def tc(self, a, alpha_target, wealth):
        assert hasattr(self, "lambda_")
        return self.lambda_ * wealth * np.abs(a - alpha_target)

    def update_wealth(self, alpha, curr_wealth, curr_pos, returns):
        assert alpha.shape[1] == returns.shape[0]
        T = returns.shape[0]
        a_grid, _ = self.set_grid()
        wealth_arr = np.zeros(T)
        for t in range(T):
            opt_alpha = float(scipy.interpolate.interp1d(a_grid, alpha[:, t], kind = "linear", fill_value = "extrapolate")(curr_pos))
            a = curr_pos * (1 + returns[t]) / (1 + (1 - curr_pos) * self.r_eff + curr_pos * returns[t])
            curr_wealth = curr_wealth * (1 + (1 - opt_alpha) * self.r_eff + opt_alpha * returns[t]) - self.tc(a, opt_alpha, curr_wealth)
            wealth_arr[t] = curr_wealth
            curr_pos = opt_alpha
        return wealth_arr, curr_pos

def main():
    dt = yf.download(tickers = "^GSPC", start = "2014-01-01", end = "2022-01-01")
    hyper_params = {"rf": .01, "zeta": -3, "N": 1000, "T": 100, "lambda_": .02, "grid": (-1, 2.5, 200)}

    strategy = DynamicStrategy(price_data = dt["Adj Close"], log_returns = False, **hyper_params)
    strategy.backtest()
    strategy.plot_pnl()

    strategy_tc = DynamicStrategyTC(price_data = dt["Adj Close"], log_returns = False, **hyper_params)
    strategy_tc.backtest()
    strategy_tc.plot_pnl()    

    strategy_grid = DynamicStrategyGrid(price_data = dt["Adj Close"], log_returns = False, **hyper_params)
    strategy_grid.backtest()
    strategy_grid.plot_pnl()

if __name__ == "__main__":
    main()
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
    
    def get_optimal_alpha(self, mu, sigma) -> float:

        # init = np.ones(2) / 2
        # obj_func = lambda x: -.5 * (np.power(1 + x@np.array([self.r_eff, mu+sigma]), self.zeta) + np.power(1 
        #             + x@np.array([self.r_eff, mu-sigma]), self.zeta))
        # constr = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # opt = scipy.optimize.minimize(fun = obj_func, x0 = init, constraints = constr)

        # idk why the previous code didn't work, but the following did. FML.
        init = .5
        obj_func = lambda x: -.5 * (np.power(1 + (1 - x)*self.r_eff + x * (mu + sigma), self.zeta) 
                                    + np.power(1 + (1 - x)*self.r_eff + x * (mu - sigma), self.zeta))
        opt = scipy.optimize.minimize(fun = obj_func, x0 = init)
        return np.array([1 - opt.x[0], opt.x[0]])
    
    def get_trade_params(self, alpha, returns):
        pnl = ((1 + alpha["riskless"] * self.r_eff + alpha["risky"] * \
               returns.shift(-1)).cumprod() - 1).shift(1).iloc[1:]
        mu_trade, sigma_trade = self.get_train_params((1 + pnl).pct_change().iloc[1:], annualize = True)
        return pd.DataFrame.from_dict({pnl.index[-1]: [mu_trade, sigma_trade, (mu_trade - self.rf) / sigma_trade]}, \
                                      orient = "index", columns = ["mu", "sigma", "Sharpe"]), pnl

    def backtest(self, verbose = 1):
        trade_eval = pd.DataFrame(columns = ["mu", "sigma", "Sharpe"])
        for i in range(self.N, self.M, self.T):
            train_set = self.returns[i - self.N : i]
            test_set = self.returns[i : i + self.T]
            mu, sigma = self.get_train_params(train_set)
            alpha_val = self.get_optimal_alpha(mu, sigma)
            alpha_df = pd.DataFrame(data = [alpha_val] * test_set.index.shape[0], index = test_set.index, columns = self.alphas.columns)

            trade_eval = pd.concat([trade_eval, self.get_trade_params(alpha_df, test_set)[0]], axis = 0)
            self.alphas = pd.concat([self.alphas, alpha_df], axis = 0)
            print(alpha_val)

        eval_overall, pnl_overall = self.get_trade_params(self.alphas, self.returns[self.N:])
        trade_eval = pd.concat([trade_eval, eval_overall.reset_index(drop = True).rename(index = {0: "overall"})], axis = 0)

        self.pnl = pnl_overall
        if verbose:
            print("Annualized mean, std dev and Sharpe Ratio: \n{}".format(trade_eval))

    def plot_pnl(self, pnl_series = None):
        assert hasattr(self, "pnl")
        if pnl_series is None:
            pnl_series = self.pnl
        plt.plot(pnl_series)
        plt.title("PnL Process of Stratgy")
        plt.xticks(rotation = 45)
        plt.xlabel("time")
        plt.ylabel("wealth")
        plt.show()

class DynamicStrategyTC(DynamicStrategy):
    def __init__(self, price_data, log_returns=False, **kwargs) -> None:
        super().__init__(price_data, log_returns, **kwargs)

    def adjust_pnl(self):
        assert hasattr(self, "lambda_")
        self.backtest(verbose = 0)
        # readability? what is that?
        changes_in_alpha = self.alphas.shift(1).iloc[np.hstack([np.arange(self.T, self.M-self.N, self.T), -1]),1].diff().shift(-1)[:-1]
        wealth_loss = ((self.pnl+1) - self.lambda_ * changes_in_alpha.abs()).dropna().cumsum().reindex(index = self.pnl.index)\
                    .ffill().fillna(0)
        pnl_adjusted = (self.pnl+1) - wealth_loss - 1
        return pnl_adjusted

    def backtest_tc(self):
        ...



def main():
    dt = yf.download(tickers = "^GSPC", start = "2014-01-01", end = "2022-01-01")
    strategy = DynamicStrategy(price_data = dt["Adj Close"], log_returns = False, rf = .01, zeta = -3, N = 1000, T = 100)
    strategy.backtest()
    # strategy.plot_pnl()

    strategy_tc = DynamicStrategyTC(price_data = dt["Adj Close"], log_returns = False, rf = .01, zeta = -3, N = 1000, T = 100, lambda_ = .02)
    adj_pnl = strategy_tc.adjust_pnl()
    strategy_tc.plot_pnl(adj_pnl)

if __name__ == "__main__":
    main()
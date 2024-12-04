import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import yule_walker

import itertools

def load_data(path, start, end):
    tickers = pd.read_csv(path).squeeze("columns").tolist()
    return yf.download(tickers, start, end)["Adj Close"].reset_index(drop = True)

class PairsTrading:
    def __init__(self, data, N, T) -> None:
        self.data = data
        self.N = N
        self.T = T
        self.get_pairs()

    def get_pairs(self):
        '''
        Description of the columns:
            C:cointegration coefficient
            status: [0, self.T - 1] if holding, -1 o.w.
            position: literally the position
            value: value of the pair in dollars.
            exit code: will be explained later.
        '''
        ticker_pairs = list(itertools.combinations(self.data.columns, 2))
        multi_index = pd.MultiIndex.from_product([range(self.N - 1, self.data.index.stop), ticker_pairs], \
                                                names = ["Day", "Pair"])
        columns = ['C', "status", "position", "value", "exit_code"]

        '''
        TODO:init all pairs on day N-1
        '''

        self.tracker = pd.DataFrame(index=multi_index, columns=columns)
        self.pairs = ticker_pairs

    def coint_test(self, stock_0: str, stock_1: str, curr_day: int):
        '''
        Test for cointegration based on the Soren Johansen Method, and the ADF test.
        return: 
            the cointegration coefficient, if there's any, else -1
        '''
        price_0 = self.data.loc[curr_day-self.N:curr_day-1, stock_0].values
        price_1 = self.data.loc[curr_day-self.N:curr_day-1, stock_1].values
        price_diff = np.diff(np.vstack([price_0, price_1]), axis = 0).T 
        reg = LinearRegression().fit(price_diff[:-1, :], price_diff[1:, :])
        lambda_, v = np.linalg.eig((reg.coef_ + np.identity(2)).T)
        lambda_idx = np.where([lambda_.imag == 0])
        if lambda_idx.shape[0] == 0:
            return -1
        c = (v[1,:] / v[0,:])[lambda_idx]
        c_pos = c[c > 0]
        if c_pos.shape[0] == 0:
            return -1
        pvals = []
        Zs = price_0 - c_pos.reshape(-1, 1)*price_1
        for i in range(c_pos.shape[0]):
            adf_test = adfuller(Zs[i, :], maxlag = 2, regression = "c", autolag = None)
            pvals.append(adf_test[1])
        opt = Zs[np.argmin(pvals), :]
        rho, sigma = yule_walker(opt, order = 2)
        return c_pos[np.argmin(pvals)] if min(pvals) <= .01 and rho.sum() <= .7 else -1
    
    def get_params(self):
        '''
        get the mu and sigma for a pair on a trading day
        '''

    def isopen(self, t, pair) -> bool:
        return self.tracker.loc[(t-1, pair), "status"] != -1
    
    def open_signal(self, Z, mu, sigma) -> int:
        '''
        given the pair price Z, mean mu, and std.dev. sigma, determine if we should 
        open a position for that pair.
        Return: the open signal
            -1 - short position
            0 - do not open
            1 - long position
        '''

    def close_signal(self, t, pair) -> int:
        '''
        given ..., determine if we should liquidate the pair.
        Return: the close signal
            0 - do not liquidate
            1 - take profit
            2 - stop loss
            3 - max holding period is reached
        '''

    def get_position(self, t, pair, C, signal):
        '''
        compute the position of a pair upon entering, given C and the open signal.
        '''

    def update_value(self, t, pair, position):
        '''
        update the value of the given pair. 
        '''

    def open(self):
        '''
        compute position, set status to 0
        '''

    def hold(self, t, pair):
        '''
        update status, update value
        '''
        self.tracker.loc[(t, pair), "status"] = self.tracker.loc[(t-1, pair), "status"] + 1

    def liquidate(self, t, pair):
        '''
        liquidate a pair. set status to -1, position to 0
        '''

    def backtest(self):
        '''
        The driver code.
        '''
        for t in range(self.N, self.data.index.stop):
            for pair in self.pairs:
                mu, sigma = self.get_params() # compute mu and sigma no matter what happens

                if self.isopen(t, pair):
                    # for opened positions
                    if self.close_signal():
                        self.liquidate(t, pair)
                    else:
                        # hold the pair
                        self.hold(t, pair)
                    self.update_value()
                else:
                    C = self.coint_test(pair[0], pair[1], t)
                    if C > 0 and (sig := self.open_signal()) != 0:
                        self.open()
                    else: 
                        ...
                        # update value using the data from the previous day.


def main():
    price_data = load_data("./TechTickers.csv", start = "2021-01-01", end = "2022-01-01")
    breakpoint()

if __name__ == "__main__":
    main()
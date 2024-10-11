import numpy as np
import pandas as pd
import yfinance as yf



def main():
    dt = yf.download(tickers = "^GSPC", start = "2014-01-01", end = "2022-01-01")
    num_days = 250
    returns = np.log1p(dt["Adj Close"].pct_change())[1:] * num_days
    rf = .01; zeta = -3; d = 2; M = returns.shape[0]; N = 1000; T = 100
    

if __name__ == "__main__":
    main()
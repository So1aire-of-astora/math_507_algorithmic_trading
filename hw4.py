import numpy as np
import pandas as pd
import yfinance as yf

def load_data(path, start, end):
    tickers = pd.read_csv(path).squeeze("columns").tolist()
    return yf.download(tickers, start, end)["Adj Close"]

class PairsTrading:
    def __init__(self) -> None:
        pass


def main():
    price_data = load_data("./TechTickers.csv")
    breakpoint()

if __name__ == "__main__":
    main()
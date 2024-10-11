import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import yfinance as yf
import scipy.optimize
from scipy import stats

#sample optimization
def my_obj(x):
    return np.sum([y**2 for y in x])
def my_constr(x):
    return np.sum(x) - 1

# constr = {"type": "eq", "fun": my_constr}
# x0 = [1,0]
# opt1 = scipy.optimize.minimize(my_obj, x0, constraints=constr, options={"maximum"})
# print(opt1)

# #sample graphs
# x = [0.1*i for i in range(100)]
# y = [math.exp(x[i]) for i in range(len(x))]
# plt.plot(x,y)
# plt.scatter([2,6],[5000,5000])
# plt.show()

# #sample regression
# x = np.random.random(1000)
# y = np.random.random(1000)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# print(intercept)
# print(slope)
# print(p_value)
# print(r_value)
# plt.scatter(x,y)

#sample data from yahoo finance
#read the list of tickers from a csv file and print them out
tickers_file = "./TechTickers.csv"

# we don't need the follwoing code actually
tickers = []
f = open(tickers_file, "r", encoding = "utf-8-sig")
for line in csv.reader(f):
    tickers.append(str(line[1]))
f.close()
tickers_str = tickers[1]
for s in tickers[2:]: 
    tickers_str = tickers_str + " " + s

# how about this
tickers_str = " ".join(pd.read_csv(tickers_file, header = None).squeeze("columns").to_list())

## data for INTC is not available
#tickers.remove(’INTC’)
#downoad the prices and volumes for the previously read list of tickers
start_date = "2013-01-01"
end_date = "2022-01-01"
stock_data = yf.download(tickers_str, start = start_date, end = end_date)

returns = stock_data["Adj Close"].pct_change().iloc[1:,:] * 250

avgs = returns.mean(axis = 0)
covs = returns.cov()

breakpoint()
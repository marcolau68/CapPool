# Idea 1: 
#   - pool based on market cap (idea is big v small)
#   - ten ranks of market cap in first pool
#   - ten subranks of market cap in second pool (or maybe another measure?)

import yfinance as yf
import pandas as pd
import numpy as np
from util import constants
import json

NUM_STOCKS = 501

# cap_dict = { t:0 for t in constants.TICKERS }

# for ticker in constants.TICKERS:
#     try:
#         cap_dict[ticker] = yf.Ticker(ticker).info["marketCap"]
#     except:
#         print(f"Problem: {ticker}")
#         continue

# with open("graph_data/market_cap.json", "w") as outfile: 
#     json.dump(cap_dict, outfile)

s_1 = np.zeros((NUM_STOCKS, 51))
s_2 = np.zeros((51, 11))

f = open("graph_data/market_cap.json")
cap_dict = json.load(f)
cap_dict = sorted(cap_dict.items(), key=lambda x:x[1])

df_1 = pd.DataFrame(data=s_1.T, columns=constants.TICKERS)

i = 0
for company in cap_dict:
    df_1[company[0]].loc[i//10] = 1
    i += 1

s_1 = df_1.to_numpy().T

for j in range(51):
    k = j // 5
    s_2[j][k] = 1

s_3 = np.matmul(s_1, s_2)


print(s_1.T.shape)
print(s_2.T.shape)

s_4 = np.matmul(s_2.T, s_1.T)
print(s_4.shape)



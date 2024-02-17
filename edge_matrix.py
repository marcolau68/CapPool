import os
import pandas as pd
import numpy as np
from dtaidistance import dtw
import util.constants as constants


def get_correlation(x, y):
    # -1 to 1
    return np.corrcoef(x, y)[0][1]

def get_dtw(x, y):
    # 0 to infinity...? 0 is identical
    return dtw.distance(x, y)

def get_covariance(x, y):
    # -inf to inf.
    return np.cov(x, y)[0][1]

def load_zscore_data():
    directory = "processed_data"
    all_data = pd.read_csv(f"{directory}/{constants.TICKERS[0]}.csv", index_col="Date")
    all_data = all_data.drop(["Close", "log_returns"], axis = 1)
    all_data = all_data.rename(columns={"log_returns_zscore": f"{constants.TICKERS[0]}_log_returns_zscore"})

    for i in range(1, len(constants.TICKERS)):
        ticker = constants.TICKERS[i]
        df = pd.read_csv(f"{directory}/{ticker}.csv", index_col="Date")
        df = df.drop(["Close", "log_returns"], axis = 1)
        df = df.rename(columns={"log_returns_zscore": f"{constants.TICKERS[0]}_log_returns_zscore"})



        # df.set_index('key').join(other.set_index('key'))

        zscores = df["log_returns_zscore"].to_numpy()
        print(ticker, zscores.shape)
        all_data.append(zscores)
       
    all_data = np.array(all_data)

    return all_data




x = np.array([1, -4, 3, 7, 5])
y = np.array([10, -3, 6, -1, -2])

all_data = load_zscore_data()




import os
import csv
import pandas as pd
import numpy as np
from dtaidistance import dtw
import util.constants as constants


def get_correlation(x, y):
    # -1 to 1
    return np.corrcoef(x, y)[0][1]

def get_covariance(x, y):
    # -inf to inf.
    return np.cov(x, y)[0][1]

# def write_zscore_data():
#     directory = "processed_data"
#     all_data = pd.read_csv(f"{directory}/{constants.TICKERS[0]}_zscores.csv", index_col="Date")
#     all_data = all_data.rename(columns={"log_returns_zscore": f"{constants.TICKERS[0]}_log_returns_zscore"})

#     for i in range(1, len(constants.TICKERS)):
#         ticker = constants.TICKERS[i]
#         df = pd.read_csv(f"{directory}/{ticker}_zscores.csv", index_col="Date")
#         df = df.rename(columns={"log_returns_zscore": f"{constants.TICKERS[i]}_log_returns_zscore"})

#         all_data = all_data.join(df)

#     # print(all_data.isna().sum())
#     all_data.fillna(value=0, inplace=True)
    
#     all_data.to_csv("graph_data/all_zscores.csv")

#     return None

def write_zscore_data(period=1):
    directory = "processed_data"
    all_data = pd.read_csv(f"{directory}/{constants.TICKERS[0]}_zscores_period={period}.csv", index_col="Date")
    all_data = all_data.rename(columns={"log_returns_zscore": f"{constants.TICKERS[0]}_log_returns_zscore"})

    for i in range(1, len(constants.TICKERS)):
        ticker = constants.TICKERS[i]
        df = pd.read_csv(f"{directory}/{ticker}_zscores.csv", index_col="Date")
        df = df.rename(columns={"log_returns_zscore": f"{constants.TICKERS[i]}_log_returns_zscore"})

        all_data = all_data.join(df)

    all_data.fillna(value=0, inplace=True)
    
    all_data.to_csv(f"graph_data/all_zscores_period={period}.csv")

    return None

def load_zscore_data():
    df = pd.read_csv("graph_data/all_zscores.csv", index_col="Date")
    return df

def get_correlation_matrix():
    data = load_zscore_data()
    n = data.shape[1]
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            col_1 = constants.TICKERS[i] + "_log_returns_zscore"
            col_2 = constants.TICKERS[j] + "_log_returns_zscore"
            
            tmp = get_correlation(data[col_1].values, data[col_2].values)
            matrix[i][j] = tmp
            matrix[j][i] = tmp

    return matrix


# Use matrix DTW and set window size for DTW
def get_dtw_matrix(window=252):
    data = load_zscore_data()
    data_np = data.to_numpy()
    data_np = data_np[-window:].transpose()
    return dtw.distance_matrix_fast(data_np)


def get_precision_matrix():
    data = load_zscore_data()
    n = data.shape[1]
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            col_1 = constants.TICKERS[i] + "_log_returns_zscore"
            col_2 = constants.TICKERS[j] + "_log_returns_zscore"
            
            tmp = get_covariance(data[col_1].values, data[col_2].values)
            matrix[i][j] = tmp
            matrix[j][i] = tmp
    
    precision_matrix = np.linalg.inv(matrix) 

    return precision_matrix


def load_edge_matrix(mode="precision"):
    if mode not in constants.EDGE_MODES:
        print("Invalid mode")
        return None
    
    edge_matrix = np.loadtxt(open(f"graph_data/{mode}_matrix.csv", "rb"), delimiter=",")
    return edge_matrix


def write_edge_matrix(mode="precision"):
    edge_matrix = None

    if mode == "correlation":
        edge_matrix = get_correlation_matrix()
    elif mode == "dtw":
        edge_matrix = get_dtw_matrix()
    elif mode == "precision":
        edge_matrix = get_precision_matrix()
    else:
        print("Invalid mode")
        return None

    np.savetxt(f"graph_data/{mode}_matrix.csv", edge_matrix, delimiter=",")

    return None




# write_edge_matrix("dtw")

# print(all_data)

write_zscore_data(period=10)



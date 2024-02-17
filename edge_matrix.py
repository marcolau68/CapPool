import os
import csv
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

def write_zscore_data():
    directory = "processed_data"
    all_data = pd.read_csv(f"{directory}/{constants.TICKERS[0]}_zscores.csv", index_col="Date")
    all_data = all_data.rename(columns={"log_returns_zscore": f"{constants.TICKERS[0]}_log_returns_zscore"})

    for i in range(1, len(constants.TICKERS)):
        ticker = constants.TICKERS[i]
        df = pd.read_csv(f"{directory}/{ticker}_zscores.csv", index_col="Date")
        df = df.rename(columns={"log_returns_zscore": f"{constants.TICKERS[i]}_log_returns_zscore"})

        all_data = all_data.join(df)

    # print(all_data.isna().sum())
    all_data.fillna(value=0, inplace=True)
    
    all_data.to_csv("graph_data/all_zscores.csv")

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

def write_correlation_matrix():
    corr_matrix = get_correlation_matrix()
    np.savetxt("graph_data/correlation_matrix.csv", corr_matrix, delimiter=",")

    return None

def load_correlation_matrix():
    corr_matrix = np.loadtxt(open("graph_data/correlation_matrix.csv", "rb"), delimiter=",")

    return corr_matrix

# Takes forever... 
def get_dtw_matrix():
    data = load_zscore_data()
    n = data.shape[1]
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            col_1 = constants.TICKERS[i] + "_log_returns_zscore"
            col_2 = constants.TICKERS[j] + "_log_returns_zscore"
            
            tmp = get_dtw(data[col_1].values, data[col_2].values)
            matrix[i][j] = tmp
            matrix[j][i] = tmp

    return matrix

def write_dtw_matrix():
    dtw_matrix = get_dtw_matrix()
    np.savetxt("graph_data/dtw_matrix.csv", dtw_matrix, delimiter=",")

    return None

def load_dtw_matrix():
    dtw_matrix = np.loadtxt(open("graph_data/dtw_matrix.csv", "rb"), delimiter=",")

    return dtw_matrix


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

def write_precision_matrix():
    precision_matrix = get_precision_matrix()
    np.savetxt("graph_data/precision_matrix.csv", precision_matrix, delimiter=",")

    return None

def load_precision_matrix():
    precision_matrix = np.loadtxt(open("graph_data/precision_matrix.csv", "rb"), delimiter=",")

    return precision_matrix



write_precision_matrix()

# print(all_data)



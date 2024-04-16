import os
import pandas as pd
import constants

# Write csv to processed_data with the ticker's raw data and its zscore
# zscore calculated with a rolling window of some period, default to 252
# assuming Gaussian, zscore standardizes data. subtract mean and divide difference by std
def process_data(ticker, window):
    df = pd.read_csv(f"raw_data/{ticker}.csv", index_col="Date")

    df["log_returns_zscore"] = (df["log_returns"] - df["log_returns"].rolling(window, min_periods=1).mean()) / df["log_returns"].rolling(window, min_periods=1).std()
    df["log_returns_zscore"].fillna(value=0, inplace=True)
    df = df.drop(["Close", "log_returns"], axis = 1)

    path = "processed_data/" + ticker + "_zscores.csv"
    df.to_csv(path)

    print(f"{ticker} Data Processed")

    return None

# Apply process_data to all files in raw_data
def process_all_data(window=252):
    directory = "raw_data"
    all_tickers = constants.TICKERS

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            ticker = f.split("/")[1].split(".")[0]
            process_data(ticker, window)
            
            if ticker not in all_tickers:
                print(ticker)
    
    return None

def var_process_data():
    all_returns = None

    for ticker in constants.TICKERS:
        filename = f"{ticker}.csv"
        f = os.path.join("raw_data", filename)

        if os.path.isfile(f):
            df = pd.read_csv(f"raw_data/{ticker}.csv", index_col="Date")
            df = df.drop(columns=["Close"]).rename(columns={"log_returns": f"{ticker}_log_returns"})

            if all_returns is None:
                all_returns = df.copy()
            else:
                all_returns[f"{ticker}_log_returns"] = df[f"{ticker}_log_returns"]
    
    all_returns.fillna(0, inplace=True)
    print(all_returns)

    path = "benchmark_data/all_log_returns.csv"
    all_returns.to_csv(path)
    
    return None
            

# ticker = "KO"
# process_data(ticker, 252)
# process_all_data()

var_process_data()







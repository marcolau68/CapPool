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
            

ticker = "KO"
# process_data(ticker, 252)

# print(stock_df)

process_all_data()








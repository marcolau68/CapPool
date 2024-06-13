import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import datetime as dt


ticker_1 = "KO"
df = pd.read_csv(f"raw_data/{ticker_1}.csv")
dates = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in df["Date"]]

raw_prices = df["Close"]
log_returns = df[np.isfinite(df['log_returns'])]["log_returns"]
zscore = (log_returns - log_returns.rolling(252).mean()) / log_returns.rolling(252).std()


period = 20
window = 20

df = df.drop(["Close", "Date"], axis=1)
df = df.rolling(period).sum()[period-1::period]


 
df["log_returns_zscore"] = (df["log_returns"] - df["log_returns"].rolling(window, min_periods=1).mean()) / df["log_returns"].rolling(window, min_periods=1).std()
df["log_returns_zscore"].fillna(value=0, inplace=True)

plt.subplot(2, 1, 1)

dates = dates[period-1::period]

plt.plot(dates, df["log_returns"], "b")
# plt.xlabel("Time")
plt.ylabel("Log Returns")
# plt.title("Coca-Cola Log Returns")
plt.title("Coca-Cola 20-Day Log Returns and Log Return Z-scores")



plt.subplot(2, 1, 2)

plt.plot(dates, df["log_returns_zscore"], "g")
plt.xlabel("Time")
plt.ylabel("Log Return Z-scores")
# plt.title("Coca-ColaLog Return Z-scores")

# plt.tight_layout()

plt.show()




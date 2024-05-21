import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import datetime as dt


ticker_1 = "AAPL"
df_1 = pd.read_csv(f"raw_data/{ticker_1}.csv")

log_returns_1 = df_1[np.isfinite(df_1['log_returns'])]["log_returns"]
zscore_1 = (log_returns_1 - log_returns_1.rolling(252).mean()) / log_returns_1.rolling(252).std()

ticker_2 = "GE"
df_2 = pd.read_csv(f"raw_data/{ticker_2}.csv")

log_returns_2 = df_2[np.isfinite(df_2['log_returns'])]["log_returns"]
zscore_2 = (log_returns_2 - log_returns_2.rolling(252).mean()) / log_returns_2.rolling(252).std()

# counts, bins = np.histogram(log_returns_1, bins=100)

# plt.stairs(counts, bins)

plt.hist(zscore_1, alpha=0.2, bins=80, label="Apple log returns z-score", density=True)
plt.hist(zscore_2, alpha=0.2, bins=80, label="General Electrics log returns z-score", density=True)
plt.legend(loc="upper left")
# plt.xlim(xmin=-0.2, xmax = 0.2)
plt.xlabel("Log Returns Z-score")
plt.ylabel("Probability")
plt.title("Apple vs General Electrics Log Returns Z-score")

plt.show()




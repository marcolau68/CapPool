import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ticker = "MMM"
df = pd.read_csv(f"processed_data/{ticker}.csv")

log_returns = df["log_returns"]
zscore = df["log_returns_zscore"]

print(len(log_returns))

# counts, bins = np.histogram(log_returns, bins=100)

# plt.stairs(counts, bins)

plt.hist(log_returns, alpha=0.2, bins=80, label="log_returns")
# plt.hist(zscore, alpha=0.2, bins=80, label="zscore")



plt.show()


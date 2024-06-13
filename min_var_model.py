import numpy as np
import pandas as pd
from util import constants
import matplotlib.pyplot as plt
import datetime as dt

df = pd.read_csv("benchmark_data/all_log_returns.csv")
dates = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in df["Date"]]
df.drop(columns=["Date"], inplace=True)

selected_columns = [ s + "_log_returns" for s in constants.EVAL_TICKERS  ] 
df = df[selected_columns]

M = 20
stock_num = df.shape[1]
n = df.shape[0]

# Adaptive min variance portfolio
rolling_mean = df.rolling(M, min_periods=1).mean().shift()
rolling_cov = df.rolling(M, min_periods=1).cov().shift()

np_rolling_cov = rolling_cov.to_numpy()

np_rolling_cov = np_rolling_cov.reshape((n, stock_num, stock_num))

inv_rolling_cov = np.zeros((n, stock_num, stock_num))

for d in range(n):
    inv_rolling_cov[d,:,:] = np.linalg.inv(np_rolling_cov[d,:,:])

weights = np.zeros((n, stock_num, 1))
ones_vec = np.ones((stock_num, 1))

for d in range(n):
    denom = np.dot(np.dot(ones_vec.T, inv_rolling_cov[d]), ones_vec)
    weights[d] = np.dot(inv_rolling_cov[d], ones_vec) / denom

weights[np.isnan(weights)] = 0
weights = weights.reshape((n, stock_num))

# Binary Classification Accuracy 
baseline_accuracy = df.gt(0).sum() / n

weights_pd = df.copy()
weights_pd[:] = weights

compare = weights_pd[constants.TRAIN_CUTOFF_INDEX:].where(weights_pd * df > 0)
accuracy = compare.count() / (n - constants.TRAIN_CUTOFF_INDEX)

print("Baseline accuracy")
print("################################")
print(baseline_accuracy)
print("################################")
print("\n")
print("Adaptive Minimum Variance Accuracy")
print("################################")
print(accuracy)
print("################################")
print("\n")


# Returns Evaluation
uniform_weights = np.ones((n, stock_num)) / stock_num
# baseline_returns = np.sum((np.exp(rolling_mean[:]) - 1) * uniform_weights, axis=1)
baseline_returns = np.sum((np.exp(df[:]) - 1) * uniform_weights, axis=1)[constants.TRAIN_CUTOFF_INDEX:]
cum_baseline_returns = np.exp(np.log(baseline_returns+1).cumsum())
baseline_std = ((np.exp(baseline_returns))).std() * pow(252, 1/2)

model_returns = np.sum((np.exp(df[:]) - 1) * weights, axis=1)[constants.TRAIN_CUTOFF_INDEX:]
cum_model_returns = np.exp(np.log(model_returns+1).cumsum())
model_std = ((np.exp(model_returns))).std() * pow(252, 1/2)

baseline_annual_returns = (np.power(cum_baseline_returns.loc[n-1], 1/6) - 1) 
model_annual_returns = (np.power(cum_model_returns.loc[n-1], 1/6) - 1)

baseline_sharpe = baseline_annual_returns / baseline_std
model_sharpe = model_annual_returns / model_std

print("Baseline")
print("################################")
print(f"Annual Returns: {round(baseline_annual_returns * 100, 2)}%")
print(f"Sharpe Ratio: {round(baseline_sharpe, 2)}")
print("################################")
print("\n")
print("Adaptive Minimum Variance Model")
print("################################")
print(f"Annual Returns: {round(model_annual_returns * 100, 2)}%")
print(f"Sharpe Ratio: {round(model_sharpe, 2)}")
print("################################")

# print(cum_model_returns)
# cum_model_returns.to_csv("graph_data/min_var_cum_returns.csv")

plt.plot(dates[constants.TRAIN_CUTOFF_INDEX:], cum_baseline_returns, label="baseline")
plt.plot(dates[constants.TRAIN_CUTOFF_INDEX:], cum_model_returns, label="model")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.ylabel("Cumulative Returns")
plt.title("Adaptive Minimum Variance vs Stock Market Cumulative Returns")
plt.show()


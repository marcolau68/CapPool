import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from util import constants
import matplotlib.pyplot as plt
import datetime as dt

# Read data
df = pd.read_csv("benchmark_data/all_log_returns.csv")
dates = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in df["Date"]]
df.drop(columns=["Date"], inplace=True)

# Set up VAR model and fit
model = VAR(df[:constants.TRAIN_CUTOFF_INDEX]) 
results = model.fit(1)

# Calculate predictions
resid = results.resid
pred_train = (df[:constants.TRAIN_CUTOFF_INDEX] - resid).fillna(0)

forecast = results.forecast(df.values[:constants.TRAIN_CUTOFF_INDEX], df.shape[0] - constants.TRAIN_CUTOFF_INDEX)
pred_test = pd.DataFrame(data=forecast, index=df[constants.TRAIN_CUTOFF_INDEX:].index, columns=df.columns)

# Isolate evaluation tickers
selected_columns = [ s + "_log_returns" for s in constants.EVAL_TICKERS ]
stock_num = len(constants.EVAL_TICKERS)

selected_resid = resid[selected_columns]
pred_train = pred_train[selected_columns]
true = df[selected_columns]
pred_test = pred_test[selected_columns]

# Calculate baseline prediction (probability of positive returns per day)
total = true.shape[0]
baseline_accuracy = true.gt(0).sum() / total

# Calculate VAR model accuracy in sample
train_total = constants.TRAIN_CUTOFF_INDEX
compare_train = pred_train.where(pred_train * true[:constants.TRAIN_CUTOFF_INDEX] > 0)
var_accuracy_train = compare_train.count() / train_total

# Calculate VAR model accuracy out of sample
test_total = df.shape[0] - constants.TRAIN_CUTOFF_INDEX
compare_test = pred_test.where(pred_test * true[constants.TRAIN_CUTOFF_INDEX:] > 0)
var_accuracy_test = compare_test.count() / test_total

# Calculate VAR MSE
mse = (selected_resid ** 2).mean()

mse_train = ((true[:constants.TRAIN_CUTOFF_INDEX] - pred_train[:constants.TRAIN_CUTOFF_INDEX]) ** 2).mean()
mse_test = ((true[constants.TRAIN_CUTOFF_INDEX:] - pred_test[constants.TRAIN_CUTOFF_INDEX:]) ** 2).mean()


# Print accuracy results
print("Baseline accuracy")
print("################################")
print(baseline_accuracy)
print("################################")
print("\n")
print("VAR Accuracy Train")
print("################################")
print(var_accuracy_train)
print("################################")
print("\n")
print("VAR Accuracy Test")
print("################################")
print(var_accuracy_test)
print("################################")
print("\n")


# Returns Evaluation
n = df.shape[0]

# Baseline
uniform_weights = np.ones((n, stock_num)) / stock_num
baseline_returns = np.sum((np.exp(true[:]) - 1) * uniform_weights, axis=1)
cum_baseline_returns = np.exp(np.log(baseline_returns+1).cumsum())
baseline_std = ((np.exp(baseline_returns))).std() * pow(252, 1/2)

baseline_annual_returns = (np.power(cum_baseline_returns.loc[n-1], 1/24) - 1) 
baseline_sharpe = baseline_annual_returns / baseline_std

# Train
train_weights = np.where(pred_train > 0, 1, -1) / 7
train_returns = np.sum((np.exp(true[:constants.TRAIN_CUTOFF_INDEX]) - 1) * train_weights, axis=1)
cum_train_returns = np.exp(np.log(train_returns+1).cumsum())

train_annual_returns = (np.power(cum_train_returns.loc[cum_train_returns.shape[0]-1], 1/18) - 1)
train_std = ((np.exp(train_returns))).std() * pow(252, 1/2)
train_sharpe = train_annual_returns / train_std

# Test
test_weights = np.where(pred_test > 0, 1, -1) / 7
test_returns = np.sum((np.exp(true[constants.TRAIN_CUTOFF_INDEX:]) - 1) * test_weights, axis=1)
cum_test_returns = np.exp(np.log(test_returns+1).cumsum())

test_annual_returns = (np.power(cum_test_returns.loc[df.shape[0]-1], 1/6) - 1)
test_std = ((np.exp(test_returns))).std() * pow(252, 1/2)
test_sharpe = test_annual_returns / test_std


# Print backtest results
print("Baseline")
print("################################")
print(f"Annual Returns: {round(baseline_annual_returns * 100, 2)}%")
print(f"Sharpe Ratio: {round(baseline_sharpe, 2)}")
print("################################")
print("\n")
print("VAR Model Train")
print("################################")
print(f"Annual Returns: {round(train_annual_returns * 100, 2)}%")
print(f"Sharpe Ratio: {round(train_sharpe, 2)}")
print("################################")
print("VAR Model Test")
print("################################")
print(f"Annual Returns: {round(test_annual_returns * 100, 2)}%")
print(f"Sharpe Ratio: {round(test_sharpe, 2)}")
print("################################")


print(test_weights[test_weights > 0].sum())
print(test_weights.shape)


plt.plot(dates[constants.TRAIN_CUTOFF_INDEX:], cum_test_returns, label="VAR test cumulative returns")
plt.plot(dates[constants.TRAIN_CUTOFF_INDEX:], cum_baseline_returns[constants.TRAIN_CUTOFF_INDEX:]/cum_baseline_returns[constants.TRAIN_CUTOFF_INDEX], label="Baseline cumulative returns")
plt.xlabel("Time")
plt.ylabel("Cumulative Returns")
plt.title("Vector Autoregressive Model Testing Cumulative Returns")
plt.legend(loc="upper left")
plt.show()
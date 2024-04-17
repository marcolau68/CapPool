import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from util import constants

# Read data
df = pd.read_csv("benchmark_data/all_log_returns.csv")
dates = df["Date"]
df.drop(columns=["Date"], inplace=True)

# Set up VAR model and fit
model = VAR(df[:constants.TRAIN_CUTOFF_INDEX]) 
results = model.fit(1)

# Calculate predictions
resid = results.resid
pred_train = (df - resid).fillna(0)

pred_test = df.copy()

forecast = results.forecast(df.values[:constants.TRAIN_CUTOFF_INDEX], df.shape[0] - constants.TRAIN_CUTOFF_INDEX)
pred_test[constants.TRAIN_CUTOFF_INDEX:] = forecast

# Isolate evaluation tickers
selected_columns = [ s + "_log_returns" for s in constants.EVAL_TICKERS ]

selected_resid = resid[selected_columns]
pred_train = pred_train[selected_columns]
true = df[selected_columns]
pred_test = pred_test[selected_columns]

# Calculate baseline prediction (probability of positive returns per day)
total = true.shape[0]
baseline_accuracy = true.gt(0).sum() / total

# Calculate VAR model accuracy in sample
train_total = constants.TRAIN_CUTOFF_INDEX
compare_train = pred_train[:constants.TRAIN_CUTOFF_INDEX].where(pred_train * true > 0)
var_accuracy_train = compare_train.count() / train_total

# Calculate VAR model accuracy out of sample
test_total = df.shape[0] - constants.TRAIN_CUTOFF_INDEX
compare_test = pred_test[constants.TRAIN_CUTOFF_INDEX:].where(pred_test * true > 0)
var_accuracy_test = compare_test.count() / test_total

# Calculate VAR MSE
mse = (selected_resid ** 2).mean()

mse_train = ((true[:constants.TRAIN_CUTOFF_INDEX] - pred_train[:constants.TRAIN_CUTOFF_INDEX]) ** 2).mean()
mse_test = ((true[constants.TRAIN_CUTOFF_INDEX:] - pred_test[constants.TRAIN_CUTOFF_INDEX:]) ** 2).mean()


# Print results
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
print("VAR Mean Squared Error Train")
print("################################")
print(mse_train)
print("################################")
print("\n")
print("VAR Mean Squared Error Test")
print("################################")
print(mse_test)
print("################################")







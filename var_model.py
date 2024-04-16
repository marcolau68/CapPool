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
results = model.fit(2)

# Calculate predictions
resid = results.resid
pred_train = (df - resid).fillna(0)

pred_test = results.forecast(df.values[constants.TRAIN_CUTOFF_INDEX-2:], 1507)
# print(pred_test)
print(pred_test.shape)

predictions_test = df.copy()
predictions_test.values[constants.TRAIN_CUTOFF_INDEX:] = pred_test
print(predictions_test[constants.TRAIN_CUTOFF_INDEX:])
print(pred_test[0])




# Isolate evaluation tickers
selected_columns = [ s + "_log_returns" for s in constants.EVAL_TICKERS ]

# selected_resid = resid[selected_columns]
# pred_train = pred_train[selected_columns]
# true = df[selected_columns]
# pred_test = pred_test[selected_columns]
# # print(pred_test)

# # Calculate baseline prediction (probability of positive returns per day)
# total = true.shape[0]
# baseline_accuracy = true.gt(0).sum() / total

# # Calculate VAR model accuracy in sample
# compare_train = pred_train.where(pred_train * true > 0)
# var_accuracy_train = compare_train.count() / total

# # Calculate VAR model accuracy out of sample
# compare_test = pred_test.where(pred_test * true > 0)
# var_accuracy_test = compare_test.count() / total

# # Calculate VAR MSE
# mse = (selected_resid ** 2).mean()

# Print results
# print("Baseline accuracy")
# print("################################")
# print(baseline_accuracy)
# print("################################")
# print("\n")
# print("VAR Accuracy")
# print("################################")
# print(var_accuracy_train)
# print(var_accuracy_test)
# print("################################")
# print("\n")
# print("VAR Mean Squared Error")
# print("################################")
# print(mse)
# print("################################")









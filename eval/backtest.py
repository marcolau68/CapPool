import numpy as np
import pandas as pd


class Backtest:
    def __init__(self):
        self.snp = None

    # Load S&P500 data?
    def load_data(self):
        return
    
    def calculate_returns(self, positions):
        self.snp["model_returns"] = self.snp["returns"] * positions

        return 
import numpy as np
import pandas as pd

class StockGraph:
    def get_nodes(self, day, window=10):
        df = pd.read_csv("graph_data/all_zscores.csv")
        df = df.drop("Date", axis=1)
        df.reindex()

        nodes = df.loc[day:day+window-1].to_numpy()
        
        return nodes
    
    def get_edges(self, mode="precision"):
        modes = ["correlation", "dtw", "precision"]

        if mode not in modes:
            print("Invalid mode")
            return None
        
        edge_matrix = np.loadtxt(open(f"graph_data/{mode}_matrix.csv", "rb"), delimiter=",")

        return edge_matrix

graph = StockGraph()
print(graph.get_nodes(day=1000, window=3))
# print(graph.get_edges("correlation").shape)


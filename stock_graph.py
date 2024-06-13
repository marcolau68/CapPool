import numpy as np
import pandas as pd
from util import constants

class StockGraph:
    def get_zscores(self, period):
        df = pd.read_csv("graph_data/all_zscores.csv")
        df = df.drop("Date", axis=1)
        df.reindex()

        df = df.rolling(period).sum()[period-1::period]

        return df
    
    def get_all_nodes(self, period=1, window=10):
        df = self.get_zscores(period)
        df.index = (df.index + 1) // 20
        features = np.zeros((df.shape[0]-window+1, window, constants.NUM_STOCKS))

        for i in range(window, df.shape[0]):
            tmp = df.loc[i-window+1:i].to_numpy()
            features[i-window] = tmp
        
        return features

    def get_nodes(self, day, period=1, window=10):
        df = self.get_zscores(period)

        # Day is the right edge of the window, df.loc is inclusive...?
        nodes = df.loc[day-period*window+1:day].to_numpy()
        
        return nodes

    def get_all_outputs(self, period=1, window=10):
        df = self.get_zscores(period)
        df.index = (df.index + 1) // 20 # becomes 1 indexed
        output = np.zeros((df.shape[0]-window, constants.NUM_STOCKS))

        for i in range(window+1, df.shape[0]):
            if bin:
                output[i-window] = df.loc[i-1] < df.loc[i]
            else:
                output[i-window] = df.loc[i]
        
        return output
    
    def get_output(self, day, period=1, bin=True):
        df = self.get_zscores(period)
        output = None

        if bin:
            output = df.loc[day-period] < df.loc[day]
            output.replace({False: -1, True: 1}, inplace=True)
        else:
            output = df.loc[day]

        return np.array(output)        
    
    def get_edge_matrix(self, mode="precision"):
        modes = ["correlation", "dtw", "precision"]

        if mode not in modes:
            print("Invalid mode")
            return None
        
        edge_matrix = np.loadtxt(open(f"graph_data/{mode}_matrix.csv", "rb"), delimiter=",")

        return edge_matrix

    def get_edge_dict(self, mode="precision", threshold=None):
        # Threshold given as upper and lower bound = [up, down]
        # Return a list of edge index [[source], [destination]]
        # Return a list of edge weights [weights]
        # Calculate threshold if not given

        edge_matrix = self.get_edge_matrix(mode)
        n, m = edge_matrix.shape

        # Set threshold to be 1 deviation above mean
        # if threshold is None:
        #     threshold = edge_matrix.mean() + edge_matrix.std()
        if threshold is None:
            if mode == "correlation":
                # Assume most correlation are positive, which is true for this case
                threshold = [edge_matrix.mean() + edge_matrix.std(), None]
            elif mode == "dtw":
                threshold = [None, edge_matrix.mean() - edge_matrix.std()]
            elif mode == "precision":
                threshold = [edge_matrix.mean() + edge_matrix.std(), edge_matrix.mean() - edge_matrix.std()]
            else:
                threshold = 0

        edge_indices = [[], []]
        edge_weights = []
        
        for i in range(n):
            for j in range(i, m):
                weight = edge_matrix[i][j]

                if i == j:
                    continue
                    
                if mode == "correlation":
                    if abs(weight) < threshold[0]:
                        continue
                elif mode == "dtw":
                    if weight > threshold[1]:
                        continue
                elif mode == "precision":
                    if weight < threshold[0] and weight > threshold[1]:
                        continue
                
                # Undirected edge
                edge_indices[0].append(i)
                edge_indices[1].append(j)
                edge_weights.append(weight)

                edge_indices[0].append(j)
                edge_indices[1].append(i)
                edge_weights.append(weight)

        return np.array(edge_indices), np.array(edge_weights)

graph = StockGraph()
nodes = graph.get_nodes(day=299, period=20, window=3)
edges = graph.get_edge_matrix()
edge_indices, edge_weights = graph.get_edge_dict(mode="precision")
out = graph.get_output(99, period=20, bin=False)
zscores = graph.get_zscores(200)

all_nodes = graph.get_all_nodes(period=20, window=3)

# print(edge_indices.shape)
# print(edge_weights.shape)
# print(nodes.shape)

print(all_nodes.shape)

# print(edges.shape)


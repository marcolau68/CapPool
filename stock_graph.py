import numpy as np
import pandas as pd

class StockGraph:
    def get_nodes(self, day, window=10):
        df = pd.read_csv("graph_data/all_zscores.csv")
        df = df.drop("Date", axis=1)
        df.reindex()

        nodes = df.loc[day:day+window-1].to_numpy()
        
        return nodes
    
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
        print(threshold)
        
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
nodes = graph.get_nodes(day=1000, window=3)
edges = graph.get_edge_matrix()

edge_indices, edge_weights = graph.get_edge_dict(mode="correlation")
print(edge_indices.shape)
print(edge_weights.shape)


# print(edges.shape)


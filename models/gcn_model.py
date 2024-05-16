import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn
from torch_geometric.nn import knn



class GNN(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, normalize=True):
    super(GNN, self).__init__()
    self.convs = torch.nn.ModuleList()
    self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
    self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
    self.convs.append(GCNConv(hidden_channels, out_channels, normalize))

  def forward(self, x, adj, mask=None):
    num_nodes, in_channels = x.size()[-2:]
    for step in range(len(self.convs)):
      # x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))
      x = F.relu(self.convs[step](x, adj, mask))
    return x
  

class Graph_Unet(torch.nn.Module):
    def __init__(self, hidden_nodes, num_features, num_nodes):
        super(Graph_Unet, self).__init__()

        # define pooling layer 1
        self.gnn1_embed = GNN(num_features, hidden_nodes, hidden_nodes)
        self.knn1 = knn()

        # define pooling layer 2
        self.gnn2_embed = GNN(hidden_nodes, hidden_nodes, hidden_nodes)

        # define pooling layer 3
        self.gnn3_embed = GNN(hidden_nodes, hidden_nodes, hidden_nodes)

        # define linear layers
        self.linear_layers = nn.Sequential(
            nn.Dropout(),
            torch.nn.Linear(hidden_nodes, num_nodes * 2),
            nn.ReLU(),
            torch.nn.Linear(num_nodes * 2, num_nodes)
        )

    # define sigmoid function for binary classification
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, adj, edge_weight=None, mask=None):
        # apply pooling layer 1
        s = self.S_1
        x = self.gnn1_embed(x, edge_index, mask)
        # x, adj, _, _ = stock_pool(x, adj, s, mask)
        x, adj, l1, e1 = knn(x, adj)
        x = x[0]
        edge_index = adj[0].nonzero().t().contiguous()

        # apply pooling layer 2
        s = self.S_2
        x = self.gnn2_embed(x, edge_index, mask)
        # x, adj, _, _ = stock_pool(x, adj, s, mask)
        x = x[0]
        edge_index = adj[0].nonzero().t().contiguous()

        # apply final convolution
        x = self.gnn3_embed(x, edge_index, mask)

        # apply linear function (leave for regression, otherwise add softmax)
        x = x.mean(dim=0)
        x = self.linear_layers(x)

        # return x # for regression
        return self.sigmoid(x)  # for binary classification

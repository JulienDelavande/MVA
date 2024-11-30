"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """GAT layer"""
    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2*n_hidden, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        
        ############## Task 1
    
        ##################
        h = self.fc(x)
        indices = adj.coalesce().indices()
        edge_features = torch.cat([h[indices[0, :]], h[indices[1, :]]], dim=1)
        e = self.leakyrelu(self.a(edge_features).squeeze())
        ##################

        attention = torch.exp(e)
        row_sum = torch.zeros(x.size(0)).to(x.device)
        row_sum.scatter_add_(0, indices[0, :], attention)
        alpha = attention / row_sum[indices[0, :]]  # alpha: [num_edges]
        #adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x.size(0), x.size(0)])).to(x.device)
        adj_att = torch.sparse_coo_tensor(indices, alpha, (x.size(0), x.size(0))).to(x.device)
        
        ##################
        out = torch.sparse.mm(adj_att, h)
        ##################

        return out, alpha


class GNN(nn.Module):
    """GNN model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        
        ############## Tasks 2 and 4
    
        ################## Tasks 2
        x, _ = self.mp1(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        x, alpha = self.mp2(x, adj)
        x = self.relu(x)
        x = self.fc(x)
        ##################

        return F.log_softmax(x, dim=1), alpha

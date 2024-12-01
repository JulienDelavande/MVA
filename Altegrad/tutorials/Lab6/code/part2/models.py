"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        ############## Task 6
    
        ##################
        A_tilde = torch.eye(adj.size(0)).to(self.device) + adj
        Z1 = self.relu(torch.mm(A_tilde, torch.mm(x_in, self.fc1.weight.T)))
        Z2 = torch.mm(A_tilde, torch.mm(Z1, self.fc2.weight.T))
        ##################

        idx = idx.unsqueeze(1).repeat(1, Z2.size(1))
        graph_representation = torch.zeros(torch.max(idx) + 1, Z2.size(1)).to(self.device)
        graph_representation = graph_representation.scatter_add_(0, idx, Z2)

        ##################
        graph_representation = self.relu(self.fc3(graph_representation))
        output = self.fc4(graph_representation)
        ##################
        
        return F.log_softmax(output, dim=1)

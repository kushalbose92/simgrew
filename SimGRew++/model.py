import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import os 
import numpy as np
import dgl
import networkx as nx

# from torch_geometric.nn import GCNConv
from gcnconv import GCNConv
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, get_laplacian
# from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(MLP, self).__init__()
        
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


# step activation function 
class custom_step_function(nn.Module):
    def __init__(self):
        super(custom_step_function, self).__init__()
    
    def forward(self, x):
        x[x>=0] = 1.0
        x[x<0] = 0.0
        return x


# defining deep gnn models
class SimGRewGCN(nn.Module):

    def __init__(self, dataset, num_layers, mlp_layers, input_dim, hidden_dim, dropout, th, alpha, device):
        super(SimGRewGCN, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.gcn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.th = th
        self.mlp_layers = mlp_layers
        self.alpha = alpha
        print("Alpha ", self.alpha)
        
        self.init_layer = nn.Linear(input_dim, hidden_dim)
        self.mlp_X = MLP(self.dataset.num_features, self.hidden_dim, self.mlp_layers, dropout=0.50).to(self.device) 
        self.step_act = custom_step_function()
        
        for i in range(self.num_layers):
            if i == 0:
                self.gcn_convs.append(GCNConv(self.dataset.num_features, self.hidden_dim).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim))
            elif i == self.num_layers - 1:
                self.gcn_convs.append(GCNConv(self.hidden_dim, self.dataset.num_classes).to(self.device))
            else:
                self.gcn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim))

        if self.th == 'None':
            self.prob = nn.Parameter(torch.rand(1, 1), requires_grad=True)
        else:
            self.prob = nn.Parameter(torch.tensor([float(self.th)]), requires_grad=True)


    def forward(self, x, adj_matrix):
        x = x.to(self.device)
        # x_0 = x
        # x = F.dropout(x, p=self.dropout, training=True)
        # x = self.init_layer(x)
        
        adj_matrix = adj_matrix.to(self.device)
        dir_energy = 0.0
    
        '''
        rewiring occurs only before fetching to the conv layers
        '''

        A_hat = self.estimate_feat_sim(x, adj_matrix, self.prob)
        updated_adj_matrix = A_hat
        total_edges = torch.count_nonzero(adj_matrix)
        # total_edges = adj_matrix.coalesce().values().shape[0]
        new_edges = torch.count_nonzero(updated_adj_matrix)
        edge_ratio = new_edges / total_edges
        
        # print(f"Original edges: {total_edges} || Current edges: {new_edges} || Edge ratio: {edge_ratio}")

        # symmetrically normalizing updated adjacency matrix
        norm_updated_adj_matrix = self.normalize_adj(updated_adj_matrix)
        scale_factor = norm_updated_adj_matrix.sum()

        # message propagation through hidden layers
        for i in range(self.num_layers):
         
            x = self.gcn_convs[i](x, norm_updated_adj_matrix)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)
                
        dir_energy = self.dirichlet_energy_with_adjacency(norm_updated_adj_matrix, x)
  
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x, dir_energy, self.prob, edge_ratio

    # generating adjacency mask 
    def estimate_feat_sim(self, X, A, prob):
        
        X_hat = self.mlp_X(X)
        A_hat = torch.matmul(X_hat, torch.t(X_hat))
        A_hat /= (torch.norm(X_hat, p = 2)**2)
        # print("Range ", A_hat.max(), "   ", A_hat.min())
        Z = torch.sub(A_hat, prob)
        # A = torch.eye(X.shape[0]).to(self.device)
        Z = Z + (self.alpha * A)
        # print("non zero in Z  ", torch.count_nonzero(Z).item())
        Z = F.relu(Z)
        # Z = torch.sigmoid(Z)
        # Z = self.step_act(Z)
        return Z

    # Dirichlet Energy wtth adjacency matrix
    def dirichlet_energy_with_adjacency(self, norm_adj, feats):
        aug_laplacian = norm_adj
        a = torch.mm(aug_laplacian, feats)
        b = torch.mm(torch.t(feats), a)
        de = torch.trace(b)
        return de
    
    # Dirichlet energy with edge indices
    def dirichlet_energy_with_edge_index(self, edge_index, feats, edge_weights):
        edge_index = edge_index.cpu()
        feats = feats.cpu()
        node_degrees = degree(edge_index[0]) + 1
        source_feats = feats[edge_index[0]]
        target_feats = feats[edge_index[1]]
        source_degrees = torch.sqrt(node_degrees[edge_index[0]])
        source_degrees = source_degrees.unsqueeze(1)
        source_degrees = source_degrees.tile((feats.shape[1],))
        target_degrees = torch.sqrt(node_degrees[edge_index[1]])
        target_degrees = target_degrees.unsqueeze(1)
        target_degrees = target_degrees.tile((feats.shape[1],))
        norm_source_feats = torch.div(source_feats, source_degrees)
        norm_target_feats = torch.div(target_feats, target_degrees)
        de = torch.sub(norm_source_feats, norm_target_feats)
        de = torch.norm(de, p = 2, dim = 1) ** 2
        if edge_weights is not None:
            edge_weights = edge_weights.cpu()
            de *= edge_weights
        de = de.sum()
        return de / 2

    # normalization of the adjacency matrix
    def normalize_adj(self, adj_matrix):
        adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0]).to(self.device)
        num_neigh = adj_matrix.sum(dim = 1, keepdim = True)
        num_neigh = num_neigh.squeeze(1)
        num_neigh = torch.sqrt(1 / num_neigh)
        degree_matrix = torch.diag(num_neigh)
        adj_matrix = torch.mm(degree_matrix, adj_matrix)
        adj_matrix = torch.mm(adj_matrix, degree_matrix)
        return adj_matrix
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
import sys

from torch_geometric.nn import GCNConv, GATConv, GCN2Conv
# from gcnconv import GCNConv
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, from_scipy_sparse_matrix, to_undirected
# from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp

# torch.autograd.set_detect_anomaly(True)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

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
class DeepGCN(nn.Module):

    def __init__(self, dataset, num_layers, mlp_layers, hidden_dim, dropout, num_nodes, th, device):
        super(DeepGCN, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.device = device
        self.gcn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.th = th
        self.mlp_layers = mlp_layers

        self.mlp_adj_mask = MLP(self.num_nodes, self.hidden_dim, self.num_nodes, self.mlp_layers, self.dropout).to(self.device)
        # self.mlp_weight = nn.Prameter(torch.FloatTensor(self.num_nodes, self.num_nodes), rquires_grad = True)
        # self.mlp_bias = nn.Prameter(torch.FloatTensor(self.num_nodes, self.num_nodes), rquires_grad = True)
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
        n_nodes = x.shape[0]
        # x = F.dropout(x, p=self.dropout, training=True)

        adj_matrix = adj_matrix.to(self.device)
        total_edges = adj_matrix.coalesce().values().shape[0]
        dir_energy = 0.0
    
        '''
        rewiring occurs only before fetching to the conv layers
        '''

        A_hat = self.estimate_adjacency_mask(adj_matrix, self.prob)
        updated_adj_matrix = A_hat
        updated_edge_index = torch.where(updated_adj_matrix != 0)
        flatten_adj = updated_adj_matrix.reshape(n_nodes * n_nodes)
        edge_weights = flatten_adj[flatten_adj != 0]
        updated_edge_index = torch.stack([updated_edge_index[0], updated_edge_index[1]])
        new_edges = len(edge_weights)

        if total_edges == 0:
            print("Empty graph")
            sys.exit()
            # edge_ratio = 0.0 
        else:
            edge_ratio = new_edges / total_edges
            

        # print(f"Original edges: {total_edges} || Current edges: {new_edges} || Edge ratio: {edge_ratio}")
        # print("prob ", self.prob)
        # print("-----------------------------")

        total_weights = torch.sum(edge_weights)
        scale_factor = (total_weights)

        # message propagation through hidden layers
        for i in range(self.num_layers):
         
            x = self.gcn_convs[i](x, updated_edge_index, edge_weights)
            
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)
                
        de = self.dirichlet_energy_with_edge_index(updated_edge_index, F.relu(x), edge_weights)
        if scale_factor != 0.0:
            dir_energy = de.item() / scale_factor.item()
  
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x, dir_energy, self.prob, edge_ratio

    # generating adjacency mask 
    def estimate_adjacency_mask(self, adj, prob):
        
        A_out = self.mlp_adj_mask(adj)
        Z = torch.sub(A_out, prob)
        Z = F.relu(Z)
        # Z = F.elu(Z)
        # Z = self.step_act(Z)
        return Z


    # Dirichlet energy with edge indices
    def dirichlet_energy_with_edge_index(self, edge_index, feats, edge_weights):
        edge_index = edge_index.cpu()
        feats = feats.cpu()
        node_degrees = degree(edge_index[0], num_nodes=feats.shape[0]) + 1
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
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

from torch_geometric.nn import GATConv, GCN2Conv
from gcnconv import GCNConv
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, get_laplacian, from_scipy_sparse_matrix
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


# SimGRew with GCN
class SimGRewGCN(nn.Module):

    def __init__(self, dataset, num_layers, mlp_layers, hidden_dim, dropout, num_nodes, th, device):
        super(SimGRewGCN, self).__init__()

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
        self.alpha = 0.0
        print("alpha ", self.alpha)
        
        self.mlp_adj_mask = MLP(self.num_nodes, self.hidden_dim, self.num_nodes, self.mlp_layers, dropout=0.50).to(self.device)
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
        # norm_adj_0 = self.normalize_adj(adj_matrix)
        # x = F.dropout(x, p=0.50, training=True)

        adj_matrix = adj_matrix.to(self.device)
        dir_energy = torch.tensor([0.0])
        edge_ratio = torch.tensor([0.0])
    
        '''
        rewiring occurs only before fetching to the conv layers
        '''

        A_hat = self.estimate_adjacency_mask(adj_matrix, self.prob)
        updated_adj_matrix = A_hat

        total_edges = adj_matrix.sum()
        # total_edges = adj_matrix.coalesce().values().shape[0]
        new_edges = torch.count_nonzero(updated_adj_matrix)
        edge_ratio = new_edges / total_edges
        # # print(f"Original edges: {total_edges} || Current edges: {new_edges} || Edge ratio: {edge_ratio}")

        # # symmetrically normalizing updated adjacency matrix
        norm_updated_adj_matrix = self.normalize_adj(updated_adj_matrix)
        # norm_weights = norm_updated_adj_matrix.sum()
        # scale_factor = (norm_weights)

        norm_updated_adj_matrix = self.normalize_adj(adj_matrix)
        # message propagation through hidden layers
        for i in range(self.num_layers):
         
            x = self.gcn_convs[i](x, norm_updated_adj_matrix)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)
                
        de = dirichlet_energy_with_adjacency(norm_updated_adj_matrix, F.relu(x))
        
        # norm_adj_matrix = norm_updated_adj_matrix.detach().cpu().numpy()
        # norm_adj_matrix_flatten = norm_adj_matrix.reshape(self.num_nodes * self.num_nodes)
        # edge_weights = norm_adj_matrix_flatten[norm_adj_matrix_flatten != 0]
        # updated_edge_index = from_scipy_sparse_matrix(sp.csr_matrix(norm_adj_matrix))[0] 
        # de = dirichlet_energy_with_edge_index(updated_edge_index, F.relu(x), edge_weights)
        
        # if norm_weights != 0.0:
        #     dir_energy = de / scale_factor
  
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x, dir_energy, self.prob, edge_ratio, norm_updated_adj_matrix
    
    # generating adjacency mask 
    def estimate_adjacency_mask(self, adj, prob):
        
        A_out = self.mlp_adj_mask(adj)
        # A_out = adj
        # A_out = torch.matmul(adj, self.mlp_weight) + self.mlp_bias
        Z = torch.sub(A_out, prob)
        # Z = A_out
        # Z += (self.alpha * adj)
        Z = F.relu(Z)
        # Z = self.step_act(Z)
        # Z = F.sigmoid(Z)
        return Z

    # normalization of the adjacency matrix
    def normalize_adj(self, adj_matrix):
        device = adj_matrix.device
        adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0]).to(device)
        num_neigh = adj_matrix.sum(dim = 1, keepdim = True)
        num_neigh = num_neigh.squeeze(1)
        num_neigh = torch.sqrt(1 / num_neigh)
        degree_matrix = torch.diag(num_neigh)
        adj_matrix = torch.mm(degree_matrix, adj_matrix)
        adj_matrix = torch.mm(adj_matrix, degree_matrix)
        # print(adj_matrix.max(), "    ", adj_matrix.min())
        return adj_matrix


# SimGRew with GAT
class SimGRewGAT(nn.Module):
    
    def __init__(self, dataset, num_layers, mlp_layers, hidden_dim, dropout, num_nodes, th, device):
        super(SimGRewGAT, self).__init__()

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
        
        heads = 2
        
        self.mlp_adj_mask = MLP(self.num_nodes, self.hidden_dim, self.num_nodes, self.mlp_layers, dropout=0.50).to(self.device)
        self.step_act = custom_step_function()
        
        for i in range(self.num_layers):
            if i == 0:
                self.gcn_convs.append(GATConv(self.dataset.num_features, self.hidden_dim, heads=heads, concat=True, add_self_loops=add_self_loops).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim*heads))
            elif i == self.num_layers - 1:
                self.gcn_convs.append(GATConv(self.hidden_dim*heads, self.dataset.num_classes, heads=heads, concat=False, add_self_loops=add_self_loops).to(self.device))
            else:
                self.gcn_convs.append(GATConv(self.hidden_dim*heads, self.hidden_dim, heads=heads, concat=True, add_self_loops=add_self_loops).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim*heads))

        if self.th == 'None':
            self.prob = nn.Parameter(torch.rand(1, 1), requires_grad=True)
        else:
            self.prob = nn.Parameter(torch.tensor([float(self.th)]), requires_grad=True)


    def forward(self, x, adj_matrix):
        x = x.to(self.device)
        # x_0 = x
        # norm_adj_0 = self.normalize_adj(adj_matrix)
        # x = F.dropout(x, p=self.dropout, training=True)

        adj_matrix = adj_matrix.to(self.device)
        dir_energy = 0.0
    
        '''
        rewiring occurs only before fetching to the conv layers
        '''

        # A_hat = self.estimate_adjacency_mask(adj_matrix, self.prob)
        # updated_adj_matrix = A_hat + torch.eye(A_hat.shape[0]).to(self.device)
        # updated_adj_matrix = A_hat
        updated_adj_matrix = adj_matrix

        total_edges = adj_matrix.sum()
        # total_edges = adj_matrix.coalesce().values().shape[0]
        new_edges = torch.count_nonzero(updated_adj_matrix)
        edge_ratio = new_edges / total_edges
        # print(f"Original edges: {total_edges} || Current edges: {new_edges} || Edge ratio: {edge_ratio}")

        # symmetrically normalizing updated adjacency matrix
        norm_updated_adj_matrix = normalize_adj(updated_adj_matrix)
        norm_weights = norm_updated_adj_matrix.sum()
        scale_factor = (norm_weights)

        norm_adj_matrix = norm_updated_adj_matrix.detach().cpu().numpy()
        norm_adj_matrix_flatten = norm_adj_matrix.reshape(self.num_nodes * self.num_nodes)
        edge_weights = norm_adj_matrix_flatten[norm_adj_matrix_flatten != 0]
        updated_edge_index = from_scipy_sparse_matrix(sp.csr_matrix(norm_adj_matrix))[0] 
        updated_edge_index = updated_edge_index.to(self.device)
        
        # print(updated_edge_index.shape)
        # message propagation through hidden layers
        for i in range(self.num_layers):
         
            x, (edge_index, edge_weights) = self.gcn_convs[i](x, updated_edge_index, return_attention_weights=True)

            if i != self.num_layers - 1:
                x = F.elu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)
                
        # de = self.dirichlet_energy_with_adjacency(norm_updated_adj_matrix, F.relu(x))
        # print(edge_weights)
        edge_weights = edge_weights[:,0].squeeze(0)
        de = dirichlet_energy_with_edge_index(updated_edge_index, F.relu(x), edge_weights)
        
        if norm_weights != 0.0:
            dir_energy = de / scale_factor
  
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x, dir_energy, self.prob, edge_ratio, norm_updated_adj_matrix
    
    # generating adjacency mask 
    def estimate_adjacency_mask(self, adj, prob):
        
        A_out = self.mlp_adj_mask(adj)
        Z = torch.sub(A_out, prob)
        # print("before  ", Z.max(), "    ", Z.min())
        Z = F.relu(Z)
        # Z = self.step_act(Z)
        # Z = F.sigmoid(Z)
        # print("after  ", Z.max(), "    ", Z.min())
        return Z



# SimGRew with GCN2Conv
class SimGRewGCN2Conv(nn.Module):
    
    def __init__(self, dataset, num_layers, mlp_layers, hidden_dim, dropout, num_nodes, th, device):
        super(SimGRewGCN2Conv, self).__init__()

        print("Model: GCNII")
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
        self.alpha = 0.3
        self.beta = 0.3
        
        self.mlp_adj_mask = MLP(self.num_nodes, self.hidden_dim, self.num_nodes, self.mlp_layers, dropout=0.50).to(self.device)
        self.step_act = custom_step_function()
        
        self.init_w = nn.Linear(self.dataset.num_features, self.hidden_dim)
        self.last_w = nn.Linear(self.hidden_dim, self.dataset.num_classes)

        for i in range(self.num_layers):
            self.gcn_convs.append(GCN2Conv(self.hidden_dim, self.alpha, self.beta, i+1))
            self.lns.append(nn.LayerNorm(self.hidden_dim))

        if self.th == 'None':
            self.prob = nn.Parameter(torch.rand(1, 1), requires_grad=True)
        else:
            self.prob = nn.Parameter(torch.tensor([float(self.th)]), requires_grad=True)


    def forward(self, x, adj_matrix):
        x = x.to(self.device)
        x = self.init_w(x)
        x_0 = x

        # norm_adj_0 = self.normalize_adj(adj_matrix)
        # x = F.dropout(x, p=self.dropout, training=True)

        adj_matrix = adj_matrix.to(self.device)
        dir_energy = 0.0
    
        '''
        rewiring occurs only before fetching to the conv layers
        '''

        # A_hat = self.estimate_adjacency_mask(adj_matrix, self.prob)
        # updated_adj_matrix = A_hat + torch.eye(A_hat.shape[0]).to(self.device)
        # updated_adj_matrix = A_hat
        updated_adj_matrix = adj_matrix

        total_edges = adj_matrix.sum()
        # total_edges = adj_matrix.coalesce().values().shape[0]
        new_edges = torch.count_nonzero(updated_adj_matrix)
        edge_ratio = new_edges / total_edges
        # print(f"Original edges: {total_edges} || Current edges: {new_edges} || Edge ratio: {edge_ratio}")

        # symmetrically normalizing updated adjacency matrix
        norm_updated_adj_matrix = normalize_adj(updated_adj_matrix)
        norm_weights = norm_updated_adj_matrix.sum()
        scale_factor = (norm_weights)

        norm_adj_matrix = norm_updated_adj_matrix.detach().cpu().numpy()
        norm_adj_matrix_flatten = norm_adj_matrix.reshape(self.num_nodes * self.num_nodes)
        edge_weights = norm_adj_matrix_flatten[norm_adj_matrix_flatten != 0]
        updated_edge_index = from_scipy_sparse_matrix(sp.csr_matrix(norm_adj_matrix))[0] 
        updated_edge_index = updated_edge_index.to(self.device)

        # message propagation through hidden layers
        for i in range(self.num_layers):
         
            x = self.gcn_convs[i](x, x_0, updated_edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)
                
        x = self.last_w(x)        
        # de = self.dirichlet_energy_with_adjacency(norm_updated_adj_matrix, F.relu(x))
        de = dirichlet_energy_with_edge_index(updated_edge_index, F.relu(x), edge_weights)
        
        if norm_weights != 0.0:
            dir_energy = de / scale_factor
  
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x, dir_energy, self.prob, edge_ratio, norm_updated_adj_matrix
    
    # generating adjacency mask 
    def estimate_adjacency_mask(self, adj, prob):
        
        A_out = self.mlp_adj_mask(adj)
        Z = torch.sub(A_out, prob)
        # print("before  ", Z.max(), "    ", Z.min())
        Z = F.relu(Z)
        # Z = self.step_act(Z)
        # Z = F.sigmoid(Z)
        # print("after  ", Z.max(), "    ", Z.min())
        return Z


# ===========================================================================================================


# Dirichlet Energy wtth adjacency matrix
def dirichlet_energy_with_adjacency(norm_adj, feats):
    device = feats.device
    aug_laplacian = torch.eye(norm_adj.shape[0]).to(device) - norm_adj
    a = torch.matmul(aug_laplacian, feats)
    b = torch.matmul(torch.t(feats), a)
    de = torch.trace(b)
    return de

# Dirichlet energy with edge indices
def dirichlet_energy_with_edge_index(edge_index, feats, edge_weights):
    feats = feats.cpu()
    edge_index = edge_index.cpu()
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
        edge_weights = torch.tensor(edge_weights)
        edge_weights = edge_weights.cpu()
        de *= edge_weights
    de = de.sum()
    return de / 2

# normalization of the adjacency matrix
def normalize_adj(adj_matrix):
    device = adj_matrix.device
    adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0]).to(device)
    num_neigh = adj_matrix.sum(dim = 1, keepdim = True)
    num_neigh = num_neigh.squeeze(1)
    num_neigh = torch.sqrt(1 / num_neigh)
    degree_matrix = torch.diag(num_neigh)
    adj_matrix = torch.mm(degree_matrix, adj_matrix)
    adj_matrix = torch.mm(adj_matrix, degree_matrix)
    # print(adj_matrix.max(), "    ", adj_matrix.min())
    return adj_matrix
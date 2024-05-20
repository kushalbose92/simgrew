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

from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
# from gcnconv import GCNConv
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, from_scipy_sparse_matrix, to_undirected, dense_to_sparse, to_dense_adj, to_dense_batch
# from torch.nn.parameter import Parameter
import torch_geometric.transforms as T 
from torch_geometric.data import Data

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp

# torch.autograd.set_detect_anomaly(True)

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
class GNN(nn.Module):

    def __init__(self, dataset, model, num_layers, mlp_layers, input_dim, hidden_dim, dropout, th, rewiring, alpha, device):
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.model = model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.gnn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.th = th
        self.mlp_layers = mlp_layers
        self.rewiring = rewiring
        self.alpha = alpha
        print("alpha ", self.alpha)
        
        # classifier layer
        self.cls = nn.Linear(self.hidden_dim, self.dataset.num_classes)


        if self.rewiring == 'simgrew':
            print("Preprocessing with SimGRew")
            self.rewiring_model = SimGRew(self.dataset.num_features, self.hidden_dim, self.mlp_layers, self.dropout, self.alpha, self.th, self.device)
        elif self.rewiring == 'fa':
            print("Adding Fully Adjacency at last layer")
            self.rewiring_model = FullyAdjacent()
        elif self.rewiring == 'all_fa':
            print("Adding Fully Adjacency at every layer")
            self.rewiring_model = FullyAdjacent()
        elif self.rewiring == 'digl':
            print("Applying Graph Diffusion as preprocessing")
            self.rewiring_model = GDCRewiring(device)
        else:
            print("No rewiring is performed")
            self.rewiring_model = None
        
        if model == 'gcn':
            print("Using GCN model...")
            for i in range(self.num_layers):
                if i == 0:
                    self.gnn_convs.append(GCNConv(self.dataset.num_features, self.hidden_dim).to(self.device))
                    self.lns.append(nn.LayerNorm(hidden_dim))
                elif i == self.num_layers - 1:
                    self.gnn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                else:
                    self.gnn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                    self.lns.append(nn.LayerNorm(hidden_dim))
        elif model == 'gin':
            print("Using GIN model...")
            for i in range(num_layers):
                if i == 0:
                    self.gnn_convs.append(GINConv(nn.Sequential(nn.Linear(self.dataset.num_features, self.hidden_dim),nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),nn.Linear(self.hidden_dim, self.hidden_dim))).to(self.device))
                else:
                    self.gnn_convs.append(GINConv(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),nn.Linear(self.hidden_dim, self.hidden_dim))).to(self.device))
        else:
            print("Invalid model name...")

        
    def forward(self, x, edge_index, batch):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        
        # x = F.dropout(x, p=self.dropout, training=True)

        dir_energy = 0.0
        edge_ratio = 0.0
        prob = torch.tensor([0.0])
            
        '''
        rewiring occurs only before fetching to the conv layers
        '''

        if self.rewiring == 'simgrew':
             rewired_edge_index, updated_edge_weights, edge_ratio, prob = self.rewiring_model.forward(x, edge_index, batch)
        elif self.rewiring == 'fa':
            rewired_edge_index, updated_edge_weights = self.rewiring_model.forward(x, edge_index, batch)
        elif self.rewiring == 'all_fa':
            rewired_edge_index, updated_edge_weights = self.rewiring_model.forward(x, edge_index, batch)
        elif self.rewiring == 'digl':
            rewired_edge_index, updated_edge_weights = self.rewiring_model.forward(x, edge_index, batch)
        elif self.rewiring == 'spectral_gap':
            rewired_edge_index, updated_edge_weights = self.rewiring_model(x, edge_index, batch)
        else:
            rewired_edge_index = edge_index
            updated_edge_weights = None

        if self.model == 'gcn':
            # message propagation through hidden layers
            for i in range(self.num_layers):

                if self.rewiring == 'fa':
                    if i != self.num_layers-1:
                        x = self.gnn_convs[i](x, edge_index, None)
                    else:
                        x = self.gnn_convs[i](x, rewired_edge_index, updated_edge_weights)
                else:
                    x = self.gnn_convs[i](x, rewired_edge_index, updated_edge_weights)

                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = self.lns[i](x)
                    x = F.dropout(x, p=self.dropout, training=True)
        else:
            for i in range(self.num_layers):
                if self.rewiring == 'fa':
                    if i != self.num_layers-1:
                        x = self.gnn_convs[i](x, edge_index)
                    else:
                        x = self.gnn_convs[i](x, rewired_edge_index)
                else:
                    x = self.gnn_convs[i](x, rewired_edge_index)
                
        # estimating DE at last layer
        dir_energy = self.dirichlet_energy_with_edge_index(rewired_edge_index, x, updated_edge_weights)
        # dir_energy /= scale_factor
        # print("last layer ", dir_energy)

        # applying mean pooling
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=True)
        x = self.cls(x)

        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x, dir_energy, prob, edge_ratio, rewired_edge_index, updated_edge_weights

    
    # Dirichlet Energy wtth adjacency matrix
    def dirichlet_energy_with_adjacency(self, norm_adj, feats):
        aug_laplacian = torch.eye(norm_adj.shape[0]).to(self.device) - norm_adj
        a = torch.mm(aug_laplacian, feats)
        b = torch.mm(torch.t(feats), a)
        de = torch.trace(b)
        return de
    
    # Dirichlet energy with edge indices
    def dirichlet_energy_with_edge_index(self, edge_index, feats, edge_weights):
        feats = feats.cpu()
        edge_index = edge_index.cpu()
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
    
        

# SimGRew rewiring technique
class SimGRew(nn.Module):
    def __init__(self, num_features, hidden_dim, mlp_layers, dropout, alpha, th, device):
        super(SimGRew, self).__init__()
        self.mlp_X = MLP(num_features, hidden_dim, mlp_layers, dropout).to(device) 
        self.step_act = custom_step_function()
        self.alpha = alpha
        self.device = device

        if th == 'None':
            self.prob = nn.Parameter(torch.rand(1, 1), requires_grad=True)
        else:
            self.prob = nn.Parameter(torch.tensor([float(th)]), requires_grad=True)

        print("Init prob ", self.prob)


    def forward(self, x, edge_index, batch):

        num_nodes = x.shape[0]
        num_graphs = max(batch).item() + 1
        _, mask = to_dense_batch(x, batch) 

        edge_batch = batch[edge_index[0].detach().cpu().numpy()]
        # print(edge_batch.shape)
        # for e in range(edge_index.shape[1]):
        #     print(edge_index[0][e], "   ", edge_index[1][e], "   ", edge_batch[e])
        
        
        adj_matrix = to_dense_adj(edge_index, batch = batch, max_num_nodes = num_nodes)
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.long)
        

        # updated_batch_adj = torch.zeros(num_graphs, num_nodes, num_nodes).to(self.device)
        rewired_edge_index = None
        updated_edge_weights = None
        
        for g in range(num_graphs):
            A_hat = self.estimate_adjacency_mask(x, adj_matrix[g], self.prob)
            graph_wise_mask = torch.tensor([1 if batch[i] == g else 0 for i in range(num_nodes)], dtype=torch.float).unsqueeze(1).to(self.device)
            num_nodes_g = torch.sum(torch.eq(batch, g))
            # print(graph_wise_mask.shape)
            graph_adj_mask = torch.matmul(graph_wise_mask, torch.t(graph_wise_mask))
            # print(graph_adj_mask)
            # for row in graph_adj_mask:
            #     print(row)
            A_hat = A_hat * graph_adj_mask
            # updated_batch_adj[g] = A_hat
            # for row in A_hat:
            #     print(row)
            # print(A_hat.shape)
            edge_indices = torch.where(A_hat != 0)
            flatten_adj = A_hat.reshape(num_nodes * num_nodes)
            edge_weights = flatten_adj[flatten_adj != 0]
            updated_edge_index = torch.stack([edge_indices[0], edge_indices[1]])
           
            if rewired_edge_index is None:
                rewired_edge_index = updated_edge_index
                updated_edge_weights = edge_weights
            else:
                rewired_edge_index = torch.cat([rewired_edge_index, updated_edge_index], dim = 1)
                updated_edge_weights = torch.cat([updated_edge_weights, edge_weights], dim = 0)
                
        
        new_edges = rewired_edge_index.shape[1]
        old_edges = edge_index.shape[1]
        edge_ratio = new_edges / old_edges

        # print(f"Original edges: {total_edges} || Current edges: {new_edges} || Edge ratio: {edge_ratio}")
        # print("-----------------------------")

        # scale_factor = new_edges

        return rewired_edge_index, updated_edge_weights, edge_ratio, self.prob

    # generating adjacency mask 
    def estimate_adjacency_mask(self, X, A, prob):
        
        X_hat = self.mlp_X(X)
        A_hat = torch.matmul(X_hat, torch.t(X_hat))
        A_hat /= (torch.norm(X_hat, p = 2)**2)
        # print("Range ", A_hat.max().item(), "   ", A_hat.min().item())
        Z = torch.sub(A_hat, prob)
        Z = Z + (self.alpha * A)
        Z = F.relu(Z)
        # Z = F.gelu(Z)
        # Z = self.step_act(Z)
        return Z


# Fully adjacency 
class FullyAdjacent(nn.Module):
    def __init__(self):
        super(FullyAdjacent, self).__init__()
        pass

    def forward(self, x, edge_index, batch):
        num_graphs = max(batch).item() + 1
        num_nodes = x.shape[0]

        adj_matrix = to_dense_adj(edge_index, batch = batch, max_num_nodes = num_nodes)
        # adj_matrix = torch.tensor(adj_matrix, dtype=torch.long)

        rewired_edge_index = None
        updated_edge_weights = None

        for g in range(num_graphs):
            fa_adj_matrix = torch.where(adj_matrix[g] == 0, 1, adj_matrix[g])
            # print(fa_adj_matrix.sum(dim=1), "  ", num_nodes)
            # fa_adj_matrix = torch.tensor(fa_adj_matrix, dtype=torch.long)
            edge_indices = torch.where(fa_adj_matrix != 0)
            flatten_adj = fa_adj_matrix.reshape(num_nodes * num_nodes)
            edge_weights = flatten_adj[flatten_adj != 0]
            updated_edge_index = torch.stack([edge_indices[0], edge_indices[1]])

            if rewired_edge_index is None:
                rewired_edge_index = updated_edge_index
                # updated_edge_weights = edge_weights
            else:
                rewired_edge_index = torch.cat([rewired_edge_index, updated_edge_index], dim = 1)
                # updated_edge_weights = torch.cat([updated_edge_weights, edge_weights], dim = 0)

        return rewired_edge_index, updated_edge_weights
        

# Diffusion improves graph learning (DIGL)
class GDCRewiring(nn.Module):
    def __init__(self, device):
        super(GDCRewiring, self).__init__()
        self.device = device

    def forward(self, x, edge_index, batch):

        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        # data = transform(data)
        # return data

        num_graphs = max(batch).item() + 1
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        edge_batch = batch[edge_index[0].detach().cpu().numpy()]
        # print(edge_batch)
        
        adj_matrix = to_dense_adj(edge_index, batch = batch, max_num_nodes = num_nodes)
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.long)
        
        rewired_edge_index = None
        updated_edge_weights = None

        # adj_list = torch.empty(num_graphs, num_nodes, num_nodes).to(self.device)
        # print("adj ", adj_matrix.shape)

        offset = 0
        prev_num_nodes_g = 0
        for g in range(num_graphs):
            node_indices = torch.where(batch == g)
            num_nodes_g = len(node_indices)
            # print(node_indices)
            x_g = x[node_indices]

            adj_matrix_g = adj_matrix[g]
            edge_index_g = dense_to_sparse(adj_matrix_g)[0]
            data_g = Data(x=x_g, edge_index=edge_index_g)
            data_tg = transform(data_g)
            edge_index_tg = data_tg.edge_index
            
            if rewired_edge_index is None:
                rewired_edge_index = edge_index_tg
                # updated_edge_weights = torch.ones_like(edge_index_tg[0])
            else:
                offset += prev_num_nodes_g
                rewired_edge_index = torch.cat([rewired_edge_index, edge_index_tg + offset], dim = 1)
                # updated_edge_weights = torch.cat([updated_edge_weights, torch.ones_like(edge_index_tg[0])], dim = 0)

            prev_num_nodes_g = num_nodes_g

        return rewired_edge_index, updated_edge_weights



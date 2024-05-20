import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import random
from torch_geometric.data import Data

import os 
import numpy as np 
from utils import *


'''
COLLAB, IMDB-BINARY, REDDIT-BINARY do not have node features
'''

class TUDatasetLoader():
    def __init__(self, dataname, transform, pre_transform):
        
        '''
        self.graphs contains the list of all input graphs
        '''
        self.graphs = TUDataset(root='TUDatasets/', name = dataname, transform = transform, pre_transform = pre_transform, pre_filter = None, use_node_attr = True, use_edge_attr = False, cleaned = False)
        # print(type(self.graphs))
        self.num_features = self.graphs.num_features
        self.num_classes = self.graphs.num_classes
        
        if dataname == 'COLLAB' or dataname == 'IMDB-BINARY' or dataname == 'REDDIT-BINARY':
            n_feats = 1
            self.graphs = list(self.graphs)
            self.pre_process(self.graphs, n_feats=n_feats)
            self.num_features = n_feats
           
        
    def pre_process(self, graphs, n_feats):
        for g in graphs:
            # print(g)
            # n = g.num_nodes
            node_degrees = degree(g.edge_index[0], num_nodes=g.num_nodes)
            # g.x = torch.ones((n, n_feats))
            # print(node_degrees.shape)

            # using node degrees as node features where nodea features are not present
            g.x = node_degrees.unsqueeze(1)
            
    def load_random_splits(self, train_split, val_split):

        num_graphs = len(self.graphs)
        indices = [i for i in range(num_graphs)]
        random.shuffle(indices)
        # print(indices)
        num_train = int(train_split*num_graphs)
        num_val = int(val_split*num_graphs)
        num_test = num_graphs - (num_train + num_val)
        train_indices = indices[:num_train]
        val_indices = indices[num_train : num_train+num_val]
        test_indices = indices[num_train+num_val:]

        # train_graphs = torch.utils.data.Subset(self.graphs, train_indices)
        # val_graphs = torch.utils.data.Subset(self.graphs, val_indices)
        # test_graphs = torch.utils.data.Subset(self.graphs, test_indices)
        
        # print(test_indices)
        # train_graphs = self.graphs[train_indices]
        # val_graphs = self.graphs[val_indices]
        # test_graphs = self.graphs[test_indices]
        
        # return train_graphs, val_graphs, test_graphs
        return train_indices, val_indices, test_indices
        
        
        
# datasets = {"reddit": reddit, "imdb": imdb, "mutag": mutag, "enzymes": enzymes, "proteins": proteins, "collab": collab}
# #datasets = {"proteins": proteins, "collab": collab}
# for key in datasets:
#     if key in ["reddit", "imdb", "collab"]:
#         for graph in datasets[key]:
#             n = graph.num_nodes
#             graph.x = torch.ones((n,1))


# [MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY]



# get spectral properties
# def get_spectral_values(A, alpha, gamma):

#     # adding parallel edges
#     A = A * gamma

#     # adding self-loops
#     A = A + (alpha * np.eye(A.shape[0]))

#     D = np.sum(A, axis=1)
#     # print(D)
#     D1 = np.diag(D)

#     # computing graph Laplacian
#     L = D1 - A

#     D2 = 1/D
#     # print(D2)
#     D2 = np.sqrt(D2)
#     D2 = np.diag(D2)
#     # print(D2)

#     # computing symmetric graph Laplacian
#     L_sym = (D2)@(L)@(D2)

#     # performing eigenvalue decomposition on L_sym
#     Z = np.linalg.eig(L_sym)
#     # print(Z)
#     # e = Z.eigenvalues
#     # v = Z.eigenvectors
#     e = Z[0]
#     v = Z[1]
#     e = np.sort(e)

#     store_e = []
#     for i in range(len(e)):
#         store_e.append(round(e[i], 4))
#         # print(store_e[i])

#     return store_e


# dataname = 'ENZYMES'
# print(dataname)
# graph_obj = TUDatasetLoader(dataname, None, None)

# g = graph_obj.graphs[5]
# print(g)

# visualize_rewired_graphs(g.edge_index, edge_weights=None, num_nodes=g.x.shape[0], data_name=dataname, num_layers=0, id=-1, flag = 'False')

# A = np.zeros((g.x.shape[0], g.x.shape[0]))
# for e in range(g.edge_index.shape[1]):
#     src = g.edge_index[0][e]
#     tgt = g.edge_index[1][e]
#     A[src][tgt] = 1

# print(A)

# print("check symmetric: ", (A - np.transpose(A)).sum())
# # print(A)

# e_0 = get_spectral_values(A, 1, 1)
# print(e_0)

# e_0 = get_spectral_values(A, 1, 3)
# print(e_0)

# A1 = np.triu(A, k=1)
# # A1=A

# print(A1)

# d1 = A1.sum(axis=1)
# print(len(d1), "  ", d1)

# e_1 = get_spectral_values(A1, 1, 1)

# # print(e_0)
# # print(e_1)

# for i in range(len(e_0)):
#     print(e_0[i], "\t", e_1[i])



# total_nodes = 0
# total_edges = 0
# count = 0
# for i, g in enumerate(graph_obj.graphs):
#     print(i, g)
#     if dataname == 'COLLAB' or 'IMDB-BINARY' or 'REDDIT-BINARY':
#         total_nodes += g.num_nodes
#     else:
#         total_nodes += g.x.shape[0]
#     total_edges += g.edge_index.shape[1]
#     count += 1
# avg_nodes = total_nodes / count 
# avg_edges = total_edges / count
# print(f'Avg #nodes: {avg_nodes} || Avg #edges : {avg_edges}')
    
# train_graphs, val_graphs, test_graphs = graph_obj.load_random_splits(0.80, 0.10)
# print(len(train_graphs), "  ", len(val_graphs), "  ", len(test_graphs))

# for i in range(len(test_graphs)):
#     print(i, "  ", test_graphs[i])


'''
DO NOT UNCOMMENT THIS PART OTHERWISE RANDOM SPLITS WILL BE LOST
'''

# ---------------- CODE FOR GENERATING RANDOM SPLITS ---------------------------------------------

# num_splits = 25

# for i in range(num_splits):
#     train_indices, val_indices, test_indices = graph_obj.load_random_splits(0.80, 0.10)
#     path = os.getcwd() + "/splits/" + dataname + "/" + dataname + "_" + str(i) + ".npz"
#     np.savez(path, arr1=train_indices, arr2=val_indices, arr3=test_indices)
    
# ------------------------------------------------------------------------------------------------

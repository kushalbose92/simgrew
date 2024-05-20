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
import seaborn as sns

from torch_geometric.utils import sort_edge_index, degree
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# visualize node embeddings
def visualize(feat_map, color, data_name, num_layers, id):
    z = TSNE(n_components=2).fit_transform(feat_map.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set1")
    if id != -1:
        plt.savefig(os.getcwd() + "/visuals/" + data_name + "_embedding_" + str(num_layers) + "_file_" + str(id) + "_.png")
    else: 
        plt.savefig(os.getcwd() + "/visuals/" + data_name + "_embedding_" + str(num_layers) + "_.png")
    plt.clf()


# loss function
def loss_fn(pred, label):
    return F.nll_loss(pred, label)

def mask_generation(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype = torch.bool)
    # print(len(index))
    mask[index] = 1
    return mask

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.unsqueeze(1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    # print(y_true.shape, "   ", y_pred.shape)

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    rocauc_list = []
    y_true = y_true.unsqueeze(1)
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


def feature_class_relation(edge_index, node_labels, feature_matrix):
    edge_index = edge_index.detach().cpu()
    node_labels = node_labels.detach().cpu()
    feature_matrix = feature_matrix.detach().cpu()

    row, col = edge_index
    row_feats = feature_matrix[row]
    col_feats = feature_matrix[col]
    
    row_labels = node_labels[row]
    col_labels = node_labels[col]

    homo_edge_count = (row_labels == col_labels).sum().int()
    hetero_edge_count = (row_labels != col_labels).sum().int()
    
    # edge_mask = torch.zeros(len(row))
    # edge_mask[row_labels == col_labels] = 1
    # edge_mask[row_labels != col_labels] = 1
   
    similarity_scores = (row_feats * col_feats).sum(dim = 1)
    # print(similarity_scores.device, "    ", edge_mask.device)
    # similarity_scores = torch.pow((row_feats - col_feats), 2).sum(dim = 1)
    # similarity_scores *= edge_mask
    total_scores = similarity_scores.sum()

    return (total_scores / (homo_edge_count + hetero_edge_count))


def degree_distribution(edge_index, num_nodes, dataname):
    node_degrees = degree(edge_index[0], num_nodes = num_nodes)
    print("Isolated nodes ", torch.sum(torch.eq(node_degrees, 0)).item())
    node_degrees = node_degrees.numpy().astype(int)
    plt.hist(node_degrees, bins=range(min(node_degrees), max(node_degrees)), alpha=0.75, color='skyblue', edgecolor='black')
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig(os.getcwd() + "/plots/" + "degree_dist_" + dataname + "_.png")
    plt.close()


# function for visualizing rewired graph
def visualize_rewired_graphs(edge_index, edge_weights, node_labels, num_nodes, data_name, num_layers, id, input_type):
    
    num_edges = edge_index.shape[1]
    G = nx.Graph()
    node_list = [i for i in range(num_nodes)]
    edge_list = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(num_edges)]
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    
    pos = nx.spring_layout(G, seed=7)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=node_labels, cmap = "Set1")

    # edges
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list, width=edge_weights, alpha=0.5, edge_color="black")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    
    if id != -1:
        if input_type == True:
            plt.savefig(os.getcwd() + "/rewired/" + data_name + "_input_graph_" + str(num_layers) + "_file_" + str(id) + "_.png")
        else:
            plt.savefig(os.getcwd() + "/rewired/" + data_name + "_rewired_graph_" + str(num_layers) + "_file_" + str(id) + "_.png")
    else: 
        if input_type == True:
            plt.savefig(os.getcwd() + "/rewired/" + data_name + "_input_graph_" + str(num_layers) + "_.png")
        else:
            plt.savefig(os.getcwd() + "/rewired/" + data_name + "_rewired_graph_" + str(num_layers) + "_.png")
            
    plt.close()
    # plt.show()
    
    
# function for visualizing rewired graph
def rewired_graphs_for_animation(edge_index, edge_weights, node_labels, num_nodes, data_name, id, val_acc):
    
    num_edges = edge_index.shape[1]
    G = nx.Graph()
    node_list = [i for i in range(num_nodes)]
    edge_list = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(num_edges)]
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    
    pos = nx.spring_layout(G, seed=7)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=node_labels, cmap = "Set1")

    # edges
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list, width=edge_weights, alpha=0.5, edge_color="black")

    # ax = plt.gca()
    # ax.margins(0.08)
    plt.axis("off")
    # plt.tight_layout()
    plt.title("[" + str(id) + " / 30] | Accuracy: " + str(round((val_acc*100), 2)), fontsize=25)
   
    plt.savefig(os.getcwd() + "/animation/" + str(id) + ".png")
    plt.close()


# class AdjRowLoader():
#     def __init__(self, dataset, train_idx, val_idx, test_idx, num_parts=10, full_epoch=False):
#         """
#         if not full_epoch, then just return one chunk of nodes
#         """
#         self.dataset = dataset
#         self.full_epoch = full_epoch
#         n = dataset.num_nodes
#         self.node_feat = dataset.x
#         self.edge_index = dataset.edge_index
#         self.edge_index = sort_edge_index(self.edge_index)
#         self.part_spots = [0]
#         self.part_nodes = [0]
#         # self.idx = idx
#         # self.mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)#, device=device)
#         # self.mask[train_idx] = True
#         self.train_mask = self.mask_generation(train_idx, dataset.num_nodes)
#         self.valid_mask = self.mask_generation(val_idx, dataset.num_nodes)
#         self.test_mask = self.mask_generation(test_idx, dataset.num_nodes)
#         num_edges = self.edge_index.shape[1]
#         approx_size = num_edges // num_parts
#         approx_part_spots = list(range(approx_size, num_edges, approx_size))[:-1]
#         for idx in approx_part_spots:
#             curr_node = self.edge_index[0,idx].item()
#             curr_idx = idx
#             while curr_idx < self.edge_index.shape[1] and self.edge_index[0,curr_idx] == curr_node:
#                 curr_idx += 1
#             self.part_nodes.append(self.edge_index[0, curr_idx].item())
#             self.part_spots.append(curr_idx)
#         self.part_nodes.append(n)
#         self.part_spots.append(self.edge_index.shape[1])
    
#     def __iter__(self):
#         self.k = 0
#         return self

#     def mask_generation(self, idx, num_nodes):
#         mask = torch.zeros(num_nodes, dtype=torch.bool)#, device=device)
#         mask[idx] = True
#         return mask
    
#     def __next__(self):
#         if self.k >= len(self.part_spots)-1:
#             raise StopIteration
            
#         if not self.full_epoch:
#             self.k = np.random.randint(len(self.part_spots)-1)
            
#         tg_data = Data()
#         batch_edge_index = self.edge_index[:, self.part_spots[self.k]:self.part_spots[self.k+1]]
#         node_ids = list(range(self.part_nodes[self.k], self.part_nodes[self.k+1]))
#         tg_data.node_ids = node_ids
#         tg_data.edge_index = batch_edge_index
#         batch_node_feat = self.node_feat[node_ids]
#         tg_data.x = batch_node_feat
#         tg_data.edge_attr = None
#         tg_data.y = self.dataset.y[node_ids]
#         tg_data.num_nodes = len(node_ids)
#         train_mask = self.train_mask[node_ids]
#         tg_data.train_mask = train_mask
#         valid_mask = self.valid_mask[node_ids]
#         tg_data.valid_mask = valid_mask
#         test_mask = self.test_mask[node_ids]
#         tg_data.test_mask = test_mask
#         self.k += 1
        
#         if not self.full_epoch:
#             self.k = float('inf')
#         return tg_data
    

def heat_map(input_adj_matrix, rewired_adj_matrix, dataname):
    fig, ax = plt.subplots(1, 2, figsize = (18, 5))
    hm1 = sns.heatmap(data = input_adj_matrix, annot = True, annot_kws={'size': 12}, linewidths = 0.1, cmap="Blues", cbar_kws={"shrink": .8}, ax=ax[0])
    hm2 = sns.heatmap(data = rewired_adj_matrix, annot = True, annot_kws={'size': 12}, linewidths = 0.1, cmap="Blues", cbar_kws={"shrink": .8}, ax=ax[1])
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Heatmap")
    plt.savefig(os.getcwd() + dataname + "_adj_heatmap.png")
    # plt.show()

def plot_details(dataname):
    prob_list = np.load(os.getcwd() + "/plots/" + dataname + "_prob_list.npy")
    de_list = np.load(os.getcwd() + "/plots/" + dataname + "_de_list.npy")
    edge_ratio_list = np.load(os.getcwd() + "/plots/" + dataname + "_edge_ratio_list.npy")

    train_iter = len(prob_list)

    plt.plot([i for i in range(train_iter)], prob_list, color='red', label='edge threshold')
    plt.xlabel('Epochs', fontsize=15)
    # plt.ylabel('')
    # plt.legend()
    plt.grid(True)
    plt.title('Edge threshold', fontsize=20)
    plt.savefig(os.getcwd() + "/plots/" + "prob_list_" + dataname + "_.png")
    plt.close()

    plt.plot([i for i in range(train_iter)], de_list, color='blue', label='Dirichlet energy')
    plt.xlabel('Epochs', fontsize=15)
    # plt.legend()
    plt.grid(True)
    plt.title('Dirichlet energy', fontsize=20)
    plt.savefig(os.getcwd() + "/plots/" + "de_list_" + dataname + "_.png")
    plt.close()

    plt.plot([i for i in range(train_iter)], edge_ratio_list, color='green', label='edge ratio')
    plt.xlabel('Epochs', fontsize=15)
    # plt.legend()
    plt.grid(True)
    plt.title('Edge ratio', fontsize=20)
    plt.savefig(os.getcwd() + "/plots/" + "edge_ratio_" + dataname + "_.png")
    plt.close()



# plot_details('snap-patents')
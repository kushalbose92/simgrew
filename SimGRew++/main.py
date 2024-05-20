import torch
import torch.nn as nn 
import numpy as np
import os
import sys
import random 

from gcnconv import GCNConv
from torch_geometric.utils import degree, remove_self_loops
from torch.nn import Parameter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from statistics import mean
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

from datacreater import *

# from model import DeepGCN
from model import *
from train import train
from test import test
from utils import visualize, mask_generation, visualize_rewired_graphs
from custom_parser import argument_parser
    

parsed_args = argument_parser().parse_args()

print(parsed_args)

dataset = parsed_args.dataset
train_lr = parsed_args.train_lr
seed = parsed_args.seed
num_layers = parsed_args.num_layers
mlp_layers = parsed_args.mlp_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
use_saved_model = parsed_args.use_saved_model
dropout = parsed_args.dropout
train_weight_decay = parsed_args.train_w_decay
th = parsed_args.th
alpha = parsed_args.alpha
device = parsed_args.device

# setting seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == 'cuda:0':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#Creating Data object
if dataset == 'Cora' or dataset == 'Citeseer' or dataset == 'Pubmed':
    data = PlanetoidData(dataset)
elif dataset == 'Chameleon' or dataset == 'Squirrel':
    data = WikipediaData(dataset)
elif dataset == 'Wisconsin' or dataset == 'Cornell' or dataset == 'Texas':
    data = WebKBData(dataset)
elif dataset == 'Film':
    data = ActorData()
else:
    print("Incorrect name of dataset")

print("Loading " + dataset)
print("No of nodes ", data.num_nodes)
print("No of edges ", data.edge_index.shape[1])
print("No of classes ", max(data.node_labels).item()+1)
print("No of features ", data.node_features.shape[1])

# data.edge_index = remove_self_loops(data.edge_index)[0]

data.node_features = data.node_features.to(device)
data.edge_index = data.edge_index.to(device)
data.node_labels = data.node_labels.to(device)


test_acc_list = []
for fid in range(10):
    print("----------------- Split " + str(fid) + " -------------------------")
    
    # adjacency matric generation
    adj_matrix = torch.zeros(data.num_nodes, data.num_nodes).to(device)
    for e in range(data.num_edges):

        src = data.edge_index[0][e]
        tgt = data.edge_index[1][e]
        adj_matrix[src][tgt] = 1
    
    f = np.load(os.getcwd() + '/splits/' + dataset.title() + '/' + dataset.lower() + '_split_0.6_0.2_'+str(fid)+'.npz')
    data.train_mask, data.val_mask, data.test_mask = f['train_mask'], f['val_mask'], f['test_mask']
    
    # data.train_mask = mask_generation(train_idx, data.num_nodes)
    # data.val_mask = mask_generation(val_idx, data.num_nodes)
    # data.test_mask = mask_generation(test_idx, data.num_nodes)
    
    # print(data.train_mask)
    
    model = SimGRewGCN(data, num_layers, mlp_layers, data.node_features.shape[1], hidden_dim, dropout, th, alpha, device)
    model = model.to(device)
    print("Init prob ", model.prob)

    init_de = model.dirichlet_energy_with_adjacency(adj_matrix, data.node_features)
    print("Initial Dirichlet Energy ", init_de, "\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr, weight_decay=train_weight_decay)

    train(data, dataset, model, adj_matrix, optimizer, train_iter, device)

    print("=" * 30)
    print("Model Testing....")

    avg_test_acc = 0.0
    model.load_state_dict(torch.load('best_gnn_model.pt'))
    model.eval()
    print("Learned threshold: ", model.prob)

    for _ in range(test_iter):
        
        test_acc, dir_energy, prob, edge_ratio = test(model, data, adj_matrix, device, is_validation = False)
        # print("Average Dirichlet Energy: ", mean(dir_energy))
        # print("Dirichlet Energy at final layer ", dir_energy[num_layers-1])
        print("Dirichlet Energy final layer ", dir_energy)
        print("Final prob: ", prob)
        print("Final edge ratio :", edge_ratio)
        # plt.plot([i for i in range(num_layers)], dir_energy, marker="*", color='blue')
        # plt.ylim(0, max(dir_energy))
        # plt.savefig(os.getcwd() + "/Dirichlet_Energy_" + str(num_layers) + ".png")
        print(f"Test Accuracy: {test_acc:.4f}")
        avg_test_acc += test_acc

    avg_test_acc = avg_test_acc / test_iter 
    print(f"Average Test Accuracy: {(avg_test_acc * 100):.4f}")

    test_acc_list.append(avg_test_acc)
    emb, pred, dir_energy,  _, _ = model(data.node_features, adj_matrix)
    visualize(emb, data.node_labels.detach().cpu().numpy(), dataset, num_layers, fid+1)
    
    # norm_adj_matrix = norm_adj_matrix.detach().cpu().numpy()
    # norm_adj_matrix_flatten = norm_adj_matrix.reshape(data.num_nodes * data.num_nodes)
    # edge_weights = norm_adj_matrix_flatten[norm_adj_matrix_flatten != 0]
    # updated_edge_index = from_scipy_sparse_matrix(sp.csr_matrix(norm_adj_matrix))[0]
    # print(updated_edge_index.shape)
    # visualize_rewired_graphs(updated_edge_index, edge_weights, data.num_nodes, dataset, num_layers, fid+1)
    
    print("---------------------------------------------\n")
    # break
    
print(test_acc_list)
print(np.average(test_acc_list)*100," || ", np.std(test_acc_list)*100)



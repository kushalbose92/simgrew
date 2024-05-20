import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import os 

from torch_geometric.loader import GraphSAINTRandomWalkSampler, RandomNodeLoader
from torch_geometric.utils import degree, remove_self_loops, add_self_loops, to_dense_adj, homophily
from torch_sparse import SparseTensor
from torch_geometric.transforms import NormalizeFeatures

import argparse
from tqdm import tqdm 
import matplotlib.pyplot as plt

import snap_patents_loader
import twitch_gamers_loader
import genius_loader 
import Penn94_loader
import arxiv_year_loader
import pokec_loader

from custom_parser import argument_parser
from model_for_large_graphs import *
# from model import *
from utils import *


parsed_args = argument_parser().parse_args()

dataname = parsed_args.dataset
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
num_parts = parsed_args.num_parts
num_splits = parsed_args.num_splits
alpha = parsed_args.alpha
# gamma = parsed_args.gamma 
device = parsed_args.device

print(parsed_args)

# setting seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == 'cuda:0':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
            

if dataname == 'snap-patents':
    dataset = snap_patents_loader.load_snap_patents_mat()
elif dataname == 'Penn94':
    dataset = Penn94_loader.load_fb100_dataset('Penn94')
elif dataname == 'pokec':
    dataset = pokec_loader.load_pokec_mat()
elif dataname == 'twitch-gamers':
    dataset = twitch_gamers_loader.load_twitch_gamer_dataset()
elif dataname == 'genius':
    dataset = genius_loader.load_genius()
elif dataname == 'arxiv-year':
    root = os.getcwd() + '/splits/arxiv-year'
    dataset = arxiv_year_loader.load_arxiv_year_dataset(root)
else:
    print("Invalid name of dataset")



dataset.num_features = dataset.x.shape[1]
dataset.num_classes = max(dataset.y).item() + 1

if dataname != 'arxiv-year' and dataname != 'snap-patents':
    dataset.edge_index = to_undirected(dataset.edge_index)  

print("----------------------------------------------")
print("Dataset: ", dataname)
print("Model Name: SimGRew++")
print("number of hidden layers:", num_layers)
print("-----------------------------------------------")
print("shape of X: ", dataset.x.shape)
print("number of nodes ", dataset.num_nodes)
print("shape of edge index ", dataset.edge_index.shape)
print("shape of y ", dataset.y.shape)
print("number of features ", dataset.num_features)
print("number of classes ", dataset.num_classes)
# print(dataset.train_mask.shape, "\t", dataset.val_mask.shape, "\t", dataset.test_mask.shape)
print("----------------------------------------------")

all_nodes = dataset.num_nodes

# node_degrees = degree(dataset.edge_index[0], num_nodes = dataset.num_nodes)
# isolated_nodes = torch.sum(torch.eq(node_degrees, 0)).item()
# print(f"Isolated nodes: {isolated_nodes} || Total nodes: {dataset.num_nodes}")

# iso_nodes_idx = torch.where(node_degrees == 0)[0]
# connected_nodes_idx = torch.where(node_degrees != 0)[0]
# print(iso_nodes_idx, "  ", iso_nodes_idx.shape, "  ", connected_nodes_idx, "  ", connected_nodes_idx.shape)

# adding self-loops to isolated nodes
# iso_edge_index = torch.stack([iso_nodes_idx, iso_nodes_idx], dim=0)
# for _ in range(int(alpha)-1):
#     iso_edge_index = add_self_loops(iso_edge_index)
#     dataset.edge_index = add_self_loops(dataset.edge_index)
# iso_edge_index = iso_edge_index.repeat(1, int(alpha))
# print(iso_edge_index.shape)

# dataset.edge_index = torch.cat([dataset.edge_index, iso_edge_index], dim=1)
# print(dataset.edge_index.shape)
# print(dataset.edge_index)


# h = homophily(dataset.edge_index, dataset.y, method='edge')
# print("before  ", h, "   ", dataset.edge_index.shape)
# dataset.edge_index = remove_self_loops(dataset.edge_index)[0]
# h = homophily(dataset.edge_index, dataset.y, method='edge')
# print("after  ", h, "    ", dataset.edge_index.shape)


# dataset.edge_index = add_self_loops(dataset.edge_index)[0]

# node_degrees = degree(dataset.edge_index[0], num_nodes = dataset.num_nodes)
# isolated_nodes = torch.sum(torch.eq(node_degrees, 0)).item()
# print(f"Isolated nodes: {isolated_nodes} || Total nodes: {dataset.num_nodes}")

# row_transform = NormalizeFeatures()
# data = Data(x = dataset.x, edge_index = dataset.edge_index)
# data = row_transform(data)
# dataset.x = data.x
# print(dataset.x.sum(dim=1))

# degree_distribution(dataset.edge_index, dataset.num_nodes, dataname)

# node_degrees = degree(dataset.edge_index[0], num_nodes = dataset.num_nodes)
# max_degree = int(torch.max(node_degrees).item())

# print(max_degree)

# for d in range(max_degree):
#     n_d = torch.sum(torch.eq(node_degrees, d)).item()
#     if n_d != 0:
#         print(f"No of nodes of degree: {d} is {n_d}")
    
# import sys 
# sys.exit()


def get_spectral_values(A, alpha, gamma):

    # adding parallel edges
    A = A * gamma

    # adding self-loops
    A = A + (alpha * np.eye(A.shape[0]))

    D = np.sum(A, axis=1)
    # print(D)
    D1 = np.diag(D)

    # computing graph Laplacian
    L = D1 - A

    D2 = 1/D
    # print(D2)
    D2 = np.sqrt(D2)
    D2 = np.diag(D2)
    # print(D2)

    # computing symmetric graph Laplacian
    L_sym = (D2)@(L)@(D2)

    # performing eigenvalue decomposition on L_sym
    Z = np.linalg.eig(L_sym)
    # print(Z)
    # e = Z.eigenvalues
    # v = Z.eigenvectors
    e = Z[0]
    v = Z[1]
    e = np.sort(e)

    store_e = []
    for i in range(len(e)):
        e_real = e[i].real
        store_e.append(round(e_real, 4))
        # print(store_e[i])

    return store_e

if dataname == 'snap-patents':
    split_idx_lst = snap_patents_loader.load_fixed_splits('snap-patents', None)
elif dataname == 'Penn94':
    split_idx_lst = Penn94_loader.load_fixed_splits('Penn94', None)
elif dataname == 'pokec':
    split_idx_lst = pokec_loader.load_fixed_splits('pokec', None)
elif dataname == 'twitch-gamers':
    split_idx_lst = twitch_gamers_loader.load_fixed_splits('twitch-gamers', None)
elif dataname == 'genius':
    split_idx_lst = genius_loader.load_fixed_splits('genius', None)
elif dataname == 'arxiv-year':
    split_idx_lst = arxiv_year_loader.load_fixed_splits("arxiv-year",os.path.join(root,"splits"))
else:
    print("Invalid name of dataset")

# print(split_idx_lst)

def mask_generation(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype = torch.bool)
    mask[index] = 1
    return mask


# train function
def train(train_loader, test_loader, model, opti_train, train_iter, model_path, device):

    best_val_acc = 0.0
    prob_list = []
    dir_energy_list = []
    edge_ratio_list = []
    for i in range(train_iter):
        print("Training iteration ", i)
        total_train_loss = 0.0
        total_val_loss = 0.0
        total_edge_ratio = 0.0
        total_prob = 0.0
        total_dir_energy = 0.0
        counter = 0

        pbar = tqdm(total = len(train_loader))
        pbar.set_description("Training")

        e_low = 0.0
        e_high = 0.0
        for _, data in enumerate(train_loader):
        
            # print(data)
            model.train()
            opti_train.zero_grad()
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            d = degree(data.edge_index[0])
            # print("isolated nodes ", torch.sum(torch.eq(d, 0)))
            # data.edge_index = remove_self_loops(data.edge_index)[0]
            # row, col = data.edge_index
            # A = SparseTensor(row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes)).to_torch_sparse_coo_tensor()
            A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            # print(A.shape, "   ", torch.nonzero(A).shape, "  ", A.sum())
            # print((A - torch.t(A)).sum())
            # A_np = A.detach().cpu().numpy()
            # eig = get_spectral_values(A_np, alpha, gamma)
            # eig = np.unique(eig)
            # for e in eig:
            #     if e < 1:
            #         e_low += 1
            #     else:
            #         e_high += 1
            # import sys
            # sys.exit()
            # print("A ", A.shape)
            emb, pred, dir_energy, prob, edge_ratio = model(data.x, A)
            label = data.y.to(device)
            pred_train = pred[data.train_mask].to(device)
            label_train = label[data.train_mask]
            loss_train = loss_fn(pred_train, label_train) 
            loss_train.backward()
            opti_train.step()

            total_train_loss += loss_train.item()
            # total_val_loss += loss_val
            total_edge_ratio += edge_ratio
            total_dir_energy += dir_energy
            total_prob += prob.item()

            pbar.update(1)
            counter += 1
        
        # print("E ", e_low, "   ", e_high)
        # break
        avg_train_loss = total_train_loss / counter
        # avg_val_loss = total_val_loss / counter
        avg_edge_ratio = total_edge_ratio / counter
        avg_dir_energy = total_dir_energy / counter
        avg_prob = total_prob / counter
            
        edge_ratio_list.append(avg_edge_ratio)
        dir_energy_list.append(avg_dir_energy)
        prob_list.append(avg_prob)

        train_acc, val_acc, test_acc = test(model, test_loader)
        
        print(f'Avg Edge Ratio: {avg_edge_ratio:.4f} || Avg Prob: {avg_prob:.4f} || Avg Dir Energy: {avg_dir_energy}')
        print(f'Iteration: {i:02d} || Train loss: {avg_train_loss: .4f} || Train Acc: {train_acc: .4f} || Valid Acc: {val_acc: .4f} || Test Acc: {test_acc: .4f}')
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)


    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot([i for i in range(train_iter)], prob_list, color='red')
    # ax[1].plot([i for i in range(train_iter)], dir_energy_list, color='blue')
    # ax[2].plot([i for i in range(train_iter)], edge_ratio_list, color='green')
    # plt.savefig("prob_de_vals_" + dataname + "_.png")
    # plt.close()


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    
    pbar = tqdm(total = len(test_loader))
    pbar.set_description("Evaluating")
  
    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    for i, data in enumerate(test_loader):
       
        # print(i, "  ", data)
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        # data.edge_index = remove_self_loops(data.edge_index)[0]
        # row, col = data.edge_index
        # A = SparseTensor(row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes)).to_torch_sparse_coo_tensor()
        A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        # print(A.shape)
        emb, pred, dir_energy, prob, edge_ratio = model(data.x, A)
        
        # print(data.y.shape)
        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(pred[mask].cpu())

        pbar.update(1)

    y_true['train'] = torch.cat(y_true['train'], dim=0)
    y_pred['train'] = torch.cat(y_pred['train'], dim=0)
    y_true['valid'] = torch.cat(y_true['valid'], dim=0)
    y_pred['valid'] = torch.cat(y_pred['valid'], dim=0)
    y_true['test'] = torch.cat(y_true['test'], dim=0)
    y_pred['test'] = torch.cat(y_pred['test'], dim=0)
    
    # print(y_true['train'].shape, "  ", y_true['valid'].shape, "  ", y_true['test'].shape)
    
    if dataname == 'genius':
        train_acc = eval_rocauc(y_true['train'], y_pred['train'])
        valid_acc = eval_rocauc(y_true['valid'], y_pred['valid'])
        test_acc = eval_rocauc(y_true['test'], y_pred['test'])
    else:
        train_acc = eval_acc(y_true['train'], y_pred['train'])
        valid_acc = eval_acc(y_true['valid'], y_pred['valid'])
        test_acc = eval_acc(y_true['test'], y_pred['test'])
            
    pbar.close()

    return train_acc, valid_acc, test_acc


print("Optimization started....")

batch_size = 4096
num_steps = 100
sample_coverage = 0
walk_length = num_layers
# processed_dir = os.getcwd() + "/data_subgraphs"

test_acc_list = []
for run in range(num_splits):
    print('')
    print(f'Split No: {run: 02d}:')

    split_idx = split_idx_lst[run]
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    train_mask = mask_generation(train_idx, all_nodes)
    valid_mask = mask_generation(valid_idx, all_nodes)
    test_mask = mask_generation(test_idx, all_nodes)

    print(train_mask.shape, " ", valid_mask.shape, " ", test_mask.shape)
    
    dataset.train_mask = train_mask 
    dataset.valid_mask = valid_mask 
    dataset.test_mask = test_mask
    
    model_path = 'best_gnn_model.pt'
    model = SimGRewGCN(dataset, num_layers, mlp_layers, dataset.x.shape[1], hidden_dim, dropout, th, alpha, device)
    model = model.to(device)
    print(model)

    opti_train = torch.optim.Adam(model.parameters(), lr = train_lr, weight_decay = train_weight_decay)
    
    # dataset = Data(x=dataset.x, edge_index = dataset.edge_index, num_classes=max(dataset.y).item() + 1, num_features = dataset.x.shape[1], y=dataset.y, train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    # sampling subgraphs for training
    train_loader = GraphSAINTRandomWalkSampler(dataset, batch_size=batch_size, walk_length=walk_length,
                                        num_steps=num_steps, sample_coverage=sample_coverage,
                                        save_dir=None)

    # # sampling subgraphs for testing
    test_loader = GraphSAINTRandomWalkSampler(dataset, batch_size=batch_size, walk_length=walk_length,
                                        num_steps=num_steps, sample_coverage=sample_coverage,
                                        save_dir=None)


    # train the model
    train(train_loader, test_loader, model, opti_train, train_iter, model_path, device)
    

    print('\n**************Evaluation**********************\n')

    model.load_state_dict(torch.load(model_path))
    for i in range(test_iter):
        train_acc, val_acc, test_acc = test(model, test_loader)
        print(f'Iteration: {i:02d}, Test Accuracy: {test_acc: .4f}')
        test_acc_list.append(test_acc)
        # visualize(out, y, data_name, num_layers)

    print("-----------------------------------------")
    # break

test_acc_list = torch.tensor(test_acc_list)
print(f'Final Test: {test_acc_list.mean():.4f} +- {test_acc_list.std():.4f}')



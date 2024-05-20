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
from utils import *


parsed_args = argument_parser().parse_args()

dataname = parsed_args.dataset
train_lr = parsed_args.train_lr
val_lr = parsed_args.val_lr
seed = parsed_args.seed
num_layers = parsed_args.num_layers
mlp_layers = parsed_args.mlp_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
de_lambda = parsed_args.de_lambda
use_saved_model = parsed_args.use_saved_model
dropout = parsed_args.dropout
train_weight_decay = parsed_args.train_w_decay
val_weight_decay = parsed_args.val_w_decay
th = parsed_args.th
num_parts = parsed_args.num_parts
num_splits = parsed_args.num_splits
device = parsed_args.device

print(parsed_args)

# settings seeds
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

print("----------------------------------------------")
print("Dataset: ", dataname)
print("Model Name: SimGRew")
print("number of hidden layers:", num_layers)
print("-----------------------------------------------")
print(dataset)
print("shape of X: ", dataset.x.shape)
print("number of nodes ", dataset.num_nodes)
print("shape of edge index ", dataset.edge_index.shape)
print("shape of y ", dataset.y.shape)
print("number of features ", dataset.num_features)
print("number of classes ", dataset.num_classes)
# print(dataset.train_mask.shape, "\t", dataset.val_mask.shape, "\t", dataset.test_mask.shape)
print("----------------------------------------------")

# dataset.edge_index = remove_self_loops(dataset.edge_index)[0]
# dataset.edge_index = add_self_loops(dataset.edge_index)[0]
# print("edge index ", dataset.edge_index.shape)

# homo_ratio = homophily(dataset.edge_index, dataset.y, method = 'edge')
# print("Homophily ratio ", homo_ratio)

# row_transform = NormalizeFeatures()
# data = Data(x = dataset.x, edge_index = dataset.edge_index)
# data = row_transform(data)
# dataset.x = data.x
# print(dataset.x.sum(dim=1))
# dataset.edge_index = remove_self_loops(dataset.edge_index)[0]
# feat_label_ratio = feature_class_relation(dataset.edge_index, dataset.y, dataset.x)
# print(f"Feature to Label ratio:  {feat_label_ratio.item(): .4f}")

# degree_distribution(dataset.edge_index, dataset.num_nodes, dataname)

# node_degrees = degree(dataset.edge_index[0], num_nodes = dataset.num_nodes)
# avg_degrees = node_degrees.sum() / dataset.num_nodes
# print(f"Avg degree: {avg_degrees}")
# import sys 
# sys.exit()


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

if dataname != 'arxiv-year' and dataname != 'snap-patents':
    dataset.edge_index = to_undirected(dataset.edge_index)  

# train function
def train(train_loader, test_loader, model, opti_train, train_iter, model_path, device):

    best_val_acc = 0.0
    prob_list = []
    de_list = []
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
        for _, data in enumerate(train_loader):
            if data.num_nodes == num_nodes_in_subgraph:
                # print("Data ", data)
                model.train()
                opti_train.zero_grad()
                data.x = data.x.to(device)
                data.edge_index = data.edge_index.to(device)
                row, col = data.edge_index
                A = SparseTensor(row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes)).to_torch_sparse_coo_tensor()
                # A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                emb, pred, dir_energy, prob, edge_ratio = model(data.x, A)
                label = data.y.to(device)
                pred_train = pred[data.train_mask].to(device)
                label_train = label[data.train_mask]
                loss_train = loss_fn(pred_train, label_train) 
                loss_train.backward()
                opti_train.step()

                # training of edge threshold t on validation loss
                # opti_val.zero_grad()
                # emb, pred, dir_energy, prob, edge_ratio = model(data.x, A)
                # # print("prob out  ", prob)
                # pred_val = pred[data.valid_mask].to(device)
                # label_val = label[data.valid_mask]
                # loss_val = loss_fn(pred_val, label_val) 
                # loss_val.backward()
                # opti_val.step()
            
                total_train_loss += loss_train.item()
                # total_val_loss += loss_val
                total_edge_ratio += edge_ratio
                total_prob += prob.item()
                total_dir_energy += dir_energy

                pbar.update(1)
                counter += 1
        
        avg_train_loss = total_train_loss / counter
        # avg_val_loss = total_val_loss / counter
        avg_edge_ratio = total_edge_ratio / counter
        avg_prob = total_prob / counter
        avg_dir_energy = total_dir_energy / counter
            
        edge_ratio_list.append(avg_edge_ratio)
        prob_list.append(avg_prob)
        de_list.append(avg_dir_energy)

        train_acc, val_acc, test_acc = test(model, test_loader)
        
        print(f'Iteration: {i:02d} || Prob: {avg_prob: .4f} || DE: {avg_dir_energy: .4f} || Edge ratio: {avg_edge_ratio: .4f}')
        print(f'Iteration: {i:02d} || Train loss: {avg_train_loss: .4f} || Train Acc: {train_acc: .4f} || Valid Acc: {val_acc: .4f} || Test Acc: {test_acc: .4f}')
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

    prob_list = np.array(prob_list)
    np.save(os.getcwd() + "/plots/" + dataname + "_prob_list.npy", prob_list)
    de_list = np.array(de_list)
    np.save(os.getcwd() + "/plots/" + dataname + "_de_list.npy", de_list)
    edge_ratio_list = np.array(edge_ratio_list)
    np.save(os.getcwd() + "/plots/" + dataname + "_edge_ratio_list.npy", edge_ratio_list)

    plot_details(dataname)


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    
    pbar = tqdm(total = len(test_loader))
    pbar.set_description("Evaluating")
  
    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    for i, data in enumerate(test_loader):
        if data.num_nodes == num_nodes_in_subgraph:
            # print(i, "  ", data)
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            row, col = data.edge_index
            A = SparseTensor(row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes)).to_torch_sparse_coo_tensor()
            # A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            # print(A)
            emb, out, dir_energy, prob, edge_ratio = model(data.x, A)
            # pred = out.argmax(dim=1)

            # print(data.y.shape)
            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(out[mask].cpu())

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


num_nodes_in_subgraph = math.floor(dataset.num_nodes / num_parts) + 1 
print("number of nodes in each subgraph ", num_nodes_in_subgraph)


print("Optimization started....")

test_acc_list = []
for run in range(num_splits):
    print('')
    print(f'Run {run:02d}:')

    split_idx = split_idx_lst[run]
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    # print(train_idx.shape, "  ", valid_idx.shape, "  ", test_idx.shape)

    
    train_mask = mask_generation(train_idx, dataset.num_nodes)
    valid_mask = mask_generation(valid_idx, dataset.num_nodes)
    test_mask = mask_generation(test_idx, dataset.num_nodes)
    dataset.train_mask = train_mask 
    dataset.valid_mask = valid_mask 
    dataset.test_mask = test_mask
    
    model_path = 'best_gnn_model_' + dataname + '_.pt'
    model = DeepGCN(dataset, num_layers, mlp_layers, hidden_dim, dropout, num_nodes_in_subgraph, th, device)
    model = model.to(device)
    print(model)
    print("Init prob ", model.prob)

    # train_params = list(model.parameters())[1:]
    # train_params = [{'params' : train_params}]
    
    # print("Printing training parameters...")
    # for params in train_params:
    #     print(params)
    
    # valid_params = list(model.parameters())[0]
    # valid_params = [{'params' : valid_params}]

    # print("Printing validation parameters...")
    # for params in valid_params:
    #     print(params)

    opti_train = torch.optim.Adam(model.parameters(), lr = train_lr, weight_decay = train_weight_decay)
    # opti_val = torch.optim.Adam(valid_params, lr = val_lr, weight_decay = val_weight_decay)
    
    # dataset = Data(x=dataset.x, edge_index = dataset.edge_index, num_classes=max(dataset.y).item() + 1, num_features = dataset.x.shape[1], y=dataset.y, train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    
    train_loader = RandomNodeLoader(dataset, num_parts = num_parts, shuffle=True, num_workers=0)
    # print("printing train loaders")
    # for i,data in enumerate(train_loader):
    #     print(i, "  ", data)
        
    test_loader = RandomNodeLoader(dataset, num_parts = num_parts, shuffle = True, num_workers=0)
    # print("printing test loaders")
    # for i,data in enumerate(test_loader):
    #     print(i, "  ", data)

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


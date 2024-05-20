import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import os 

from torch_geometric.loader import GraphSAINTRandomWalkSampler, RandomNodeLoader
from torch_geometric.utils import degree, remove_self_loops, add_self_loops, to_dense_adj
from torch_sparse import SparseTensor

import matplotlib.pyplot as plt
from genius_loader import *

import argparse
from tqdm import tqdm 

from custom_parser import argument_parser
from model_for_large_graphs import *
from utils import *


parsed_args = argument_parser().parse_args()

dataset = parsed_args.dataset
train_lr = parsed_args.train_lr
val_lr = parsed_args.val_lr
seed = parsed_args.seed
num_layers = parsed_args.num_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
de_lambda = parsed_args.de_lambda
use_saved_model = parsed_args.use_saved_model
dropout = parsed_args.dropout
train_weight_decay = parsed_args.train_w_decay
val_weight_decay = parsed_args.val_w_decay
th = parsed_args.th
device = parsed_args.device

print("Device: ", device)

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if device == 'cuda:0':
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
            

dataset = load_genius()
dataset.num_features = dataset.x.shape[1]
dataset.num_classes = max(dataset.y).item() + 1

print("----------------------------------------------")
print("Dataset: genius")
print("Model Name: Dirichlet Energy-constrained Graph Rewiring")
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


num_steps = 5
split_idx_lst = load_fixed_splits('genius', None)
# print(split_idx_lst)

# train function
def train(train_loader, test_loader, model, opti_train, opti_val, de_lambda, train_iter, model_path, device):

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
                # print(data)
                model.train()
                opti_train.zero_grad()
                # print("prob before ", model.prob)
                data.x = data.x.to(device)
                data.edge_index = data.edge_index.to(device)
                row, col = data.edge_index
                A = SparseTensor(row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes)).to_torch_sparse_coo_tensor()
                # A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                # print(A)
                emb, pred, dir_energy, prob, edge_ratio = model(data.x, A)
                label = data.y.to(device)
                pred_train = pred[data.train_mask].to(device)
                label_train = label[data.train_mask]
                loss_train = loss_fn(pred_train, label_train) 
                # de_loss = (dir_energy[0] * 1.0)
                # loss_train += de_loss
                loss_train.backward()
                opti_train.step()

                # training of edge threshold t on validation loss
                opti_val.zero_grad()
                emb, pred, dir_energy, prob, edge_ratio = model(data.x, A)
                # prob_list.append(prob.item())
                pred_val = pred[data.valid_mask].to(device)
                label_val = label[data.valid_mask]
                loss_val = loss_fn(pred_val, label_val) 
                # de_loss = (dir_energy[-1] * de_lambda)
                # loss_val += de_loss
                loss_val.backward()
                opti_val.step()
            
                total_train_loss += loss_train
                total_val_loss += loss_val
                total_edge_ratio += edge_ratio
                total_prob += prob
                # total_dir_energy += dir_energy[-1]

                pbar.update(1)
                counter += 1
        
        
        if counter != 0:
            avg_train_loss = total_train_loss / counter
            avg_val_loss = total_val_loss / counter
            avg_edge_ratio = total_edge_ratio / counter
            avg_prob = total_prob / counter
            # avg_dir_energy = total_dir_energy / counter
            
            # de_list.append(avg_dir_energy)
            edge_ratio_list.append(avg_edge_ratio)
            prob_list.append(avg_prob.item())

        train_acc, val_acc, test_acc = test(model, test_loader)
        print(f'Iteration: {i:02d} || Train Acc: {train_acc: .4f} || Valid Acc: {val_acc: .4f} || Test Acc: {test_acc: .4f}')
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)


    fig, ax = plt.subplots(2, 1)
    ax[0].plot([i for i in range(train_iter)], prob_list, color='red')
    # ax[1].plot([i for i in range(counter)], de_list, color='blue')
    # ax[2].plot([i for i in range(train_iter)], de_adj_list, color='brown')
    ax[1].plot([i for i in range(train_iter)], edge_ratio_list, color='green')
    plt.savefig("prob_de_vals_genius.png")
    plt.close()


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
    
    train_acc = eval_rocauc(y_true['train'], y_pred['train'])
    valid_acc = eval_rocauc(y_true['valid'], y_pred['valid'])
    test_acc = eval_rocauc(y_true['test'], y_pred['test'])
            
    pbar.close()

    return train_acc, valid_acc, test_acc


num_parts = 100
num_nodes_in_subgraph = math.floor(dataset.num_nodes / num_parts) + 1 
print("number of nodes in each subgraph ", num_nodes_in_subgraph)


print("Optimization started....")

test_acc_list = []
for run in range(num_steps):
    print('')
    print(f'Run {run:02d}:')
    
    model_path = 'best_gnn_model.pt'
    model = DeepGCN(dataset, num_layers, hidden_dim, dropout, num_nodes_in_subgraph, th, device)
    model = model.to(device)
    # print(model)
    print("Init prob ", model.prob)


    train_params = list(model.parameters())[1:]
    train_params = [{'params' : train_params}]
    # print(train_params)
    # for params in train_params:
    #     print(params.shape)
    valid_params = list(model.parameters())[0]
    valid_params = [{'params' : valid_params}]

    # for params in valid_params:
    #     print(params.shape)

    opti_train = torch.optim.Adam(train_params, lr = train_lr, weight_decay = train_weight_decay)
    opti_val = torch.optim.Adam(valid_params, lr = val_lr, weight_decay = val_weight_decay)

    split_idx = split_idx_lst[run]
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    train_mask = mask_generation(train_idx, dataset.num_nodes)
    valid_mask = mask_generation(valid_idx, dataset.num_nodes)
    test_mask = mask_generation(test_idx, dataset.num_nodes)
    dataset.train_mask = train_mask 
    dataset.valid_mask = valid_mask 
    dataset.test_mask = test_mask
    
    train_loader = RandomNodeLoader(dataset, num_parts = num_parts, shuffle=True, num_workers=5)
    # print("printing train loaders")
    # for i,data in enumerate(train_loader):
    #     print(i, "  ", data)
    test_loader = RandomNodeLoader(dataset, num_parts = num_parts, shuffle = False, num_workers=5)
    # print("printing test loaders")
    # for i,data in enumerate(test_loader):
    #     print(i, "  ", data)

    # train the model
    train(train_loader, test_loader, model, opti_train, opti_val, de_lambda, train_iter, model_path, device)

    print('\n**************Evaluation**********************\n')

    model.load_state_dict(torch.load(model_path))
    for i in range(test_iter):
        train_acc, val_acc, test_acc = test(model, test_loader)
        print(f'Iteration: {i:02d}, Test Accuracy: {test_acc: .4f}')
        test_acc_list.append(test_acc)
        # visualize(out, y, data_name, num_layers)

    print("-----------------------------------------")
    break

test_acc_list = torch.tensor(test_acc_list)
print(f'Final Test: {test_acc_list.mean():.4f} +- {test_acc_list.std():.4f}')



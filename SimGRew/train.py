import torch
import torch.nn as nn 
import numpy as np
import os

import matplotlib.pyplot as plt 
from statistics import mean
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse as sp 

from utils import *
from test import *


# train function
def train(dataset, dataname, model, adj_matrix, opti_train, de_lambda, train_iter, model_path, device):

    best_val_acc = 0.0
    prob_list = []
    de_list = []
    # de_feats_list = []
    # de_adj_list = []
    edge_ratio_list = []
    counter = 0
    for i in range(train_iter):

        # training of gnn model weights on training loss
        model.train()
        opti_train.zero_grad()
        # print("prob before ", model.prob)
        emb, pred, dir_energy, prob, edge_ratio, norm_adj_matrix = model(dataset.node_features, adj_matrix)
        label = dataset.node_labels.to(device)
        pred_train = pred[dataset.train_mask].to(device)
        label_train = label[dataset.train_mask]
        loss_train = loss_fn(pred_train, label_train) 
        # de_loss = (dir_energy * de_lambda)
        # loss_train += de_loss
        loss_train.backward()
        opti_train.step()

        # training of edge threshold t on validation loss
        # opti_val.zero_grad()
        # emb, pred, dir_energy, prob, edge_ratio, _ = model(dataset.node_features, adj_matrix)
        # pred_val = pred[dataset.val_mask].to(device)
        # label_val = label[dataset.val_mask]
        # loss_val = loss_fn(pred_val, label_val) 
        # # de_loss = (dir_energy * de_lambda)
        # # loss_val += de_loss
        # loss_val.backward()
        # opti_val.step()
        loss_val = 0.0

        val_acc, _, _, _, _ = test(model, dataset, adj_matrix, device, is_validation = True)


        # if i == 0 or (i+1) % 5 == 0:
        #     norm_adj_matrix = norm_adj_matrix.detach().cpu().numpy()
        #     norm_adj_matrix_flatten = norm_adj_matrix.reshape(dataset.num_nodes * dataset.num_nodes)
        #     edge_weights = norm_adj_matrix_flatten[norm_adj_matrix_flatten != 0]
        #     updated_edge_index = from_scipy_sparse_matrix(sp.csr_matrix(norm_adj_matrix))[0]
        #     rewired_graphs_for_animation(updated_edge_index.cpu(), edge_weights, dataset.node_labels.cpu(), dataset.num_nodes, dataname, counter, val_acc)
        #     counter += 1
        #     print(i, "-th rewired graph plotted")
            
        prob_list.append(prob.item())
        de_list.append(dir_energy.item())
        edge_ratio_list.append(edge_ratio.item())
        
        # print(f'prob: {prob.item():.4f} || edge ratio: {edge_ratio.item():.4f} || DE: {dir_energy[-1].item():.4f}')
        
        if i % 100 == 0:
            print(f"Epoch: {i+1:03d}, Train Loss: {loss_train:.4f}, Val Loss: {loss_val:.4f}, Val Acc: {val_acc:.4f}")
            print(f"DE : {dir_energy.item():.4f} || Edge ratio: {edge_ratio.item():.4f} || Prob: {prob.item():.4f}")
            
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

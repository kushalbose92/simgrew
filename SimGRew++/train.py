import torch
import torch.nn as nn 
import numpy as np
import os

import matplotlib.pyplot as plt 
from statistics import mean

from utils import *
from test import test


# train function
def train(dataset, dataname, model, adj_matrix, opti, train_iter, device):

    best_val_acc = 0.0
    prob_list = []
    de_list = []
    edge_ratio_list = []
    for i in range(train_iter):

        # training of gnn model weights on training loss
        model.train()
        opti.zero_grad()
        # print("prob before ", model.prob)
        emb, pred, dir_energy, prob, edge_ratio = model(dataset.node_features, adj_matrix)
        label = dataset.node_labels.to(device)
        pred_train = pred[dataset.train_mask].to(device)
        label_train = label[dataset.train_mask]
        loss_train = loss_fn(pred_train, label_train) 
        # de_loss = (dir_energy[0] * 1.0)
        # loss_train += de_loss
        loss_train.backward()
        opti.step()
        # de_feats_list.append(dir_energy[0].item())

        val_acc, _, _, _ = test(model, dataset, adj_matrix, device, is_validation = True)

        prob_list.append(prob.item())
        de_list.append(dir_energy.item())
        edge_ratio_list.append(edge_ratio.item())
        
        if i % 100 == 0:
            print(f"Epoch: {i+1:03d}, Train Loss: {loss_train:.4f}, Val Acc: {val_acc:.4f}")
            print(f"DE : {dir_energy.item():.4f} ||  Edge ratio {edge_ratio.item():.4f} || Prob {prob.item():.4f}")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gnn_model.pt')

    prob_list = np.array(prob_list)
    np.save(os.getcwd() + "/plots/" + dataname + "_prob_list.npy", prob_list)
    de_list = np.array(de_list)
    np.save(os.getcwd() + "/plots/" + dataname + "_de_list.npy", de_list)
    edge_ratio_list = np.array(edge_ratio_list)
    np.save(os.getcwd() + "/plots/" + dataname + "_edge_ratio_list.npy", edge_ratio_list)

    plot_details(dataname)



    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot([i for i in range(train_iter)], prob_list, color='red')
    # ax[1].plot([i for i in range(train_iter)], de_list, color='blue')
    # # ax[2].plot([i for i in range(train_iter)], de_adj_list, color='brown')
    # ax[2].plot([i for i in range(train_iter)], edge_ratio_list, color='green')
    # plt.savefig("prob_de_vals.png")
    # plt.close()
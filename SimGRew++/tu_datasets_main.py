import torch
import torch.nn as nn 
import numpy as np
import os
import sys
import random 
from tqdm import tqdm

from gcnconv import GCNConv
from torch_geometric.utils import degree, remove_self_loops, to_dense_adj, to_dense_batch, dense_to_sparse

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.nn import Parameter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from statistics import mean

from tu_datasets_loader import *
from model_graph_classification import *
from utils import *
from custom_parser import argument_parser

import sdrf
import fosr
import borf
from gtr import PrecomputeGTREdges, AddPrecomputedGTREdges

    
torch.autograd.set_detect_anomaly(True)

# train function
def train(train_loader, val_loader, test_loader, model, opti_train, train_iter, model_path, device):

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
            # print("Data ", data)
            model.train()
            opti_train.zero_grad()
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            # print(data.num_nodes)
            # print(data.batch)
            # A = to_dense_adj(data.edge_index, batch = data.batch, max_num_nodes=data.num_nodes)[0]
            # A = torch.tensor(A, dtype=torch.long)
            # print(A[0][:5])
            # print(A.shape)
            emb, pred, dir_energy, prob, edge_ratio, _, _ = model(data.x, data.edge_index, data.batch)
            data.y = data.y.to(device)
            loss_train = loss_fn(pred, data.y) 
            loss_train.backward()
            opti_train.step()
        
            total_train_loss += loss_train.item()
            total_edge_ratio += edge_ratio
            total_prob += prob.item()
            total_dir_energy += dir_energy

            pbar.update(1)
            counter += 1
        
        avg_train_loss = total_train_loss / counter
        avg_edge_ratio = total_edge_ratio / counter
        avg_prob = total_prob / counter
        avg_dir_energy = total_dir_energy / counter
            
        edge_ratio_list.append(avg_edge_ratio)
        prob_list.append(avg_prob)
        de_list.append(avg_dir_energy.item())

        val_acc, _, _, _ = test(model, val_loader, flag = 'Valid')
        
        print(f'Iteration: {i:02d} || Prob: {avg_prob: .4f} || DE: {avg_dir_energy: .4f} || Edge ratio: {avg_edge_ratio: .4f}')
        print(f'Iteration: {i:02d} || Train loss: {avg_train_loss: .4f} || Valid Acc: {val_acc: .4f}')
    
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


# test function
@torch.no_grad()
def test(model, loader, flag):
    model.eval()
    correct = 0

    pbar = tqdm(total = len(loader))
    pbar.set_description("Evaluating")
    for i, data in enumerate(loader):
        # print(data)
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        # print(data.num_nodes)
        # print(data.batch)
        # A = to_dense_adj(data.edge_index, batch = data.batch, max_num_nodes=data.num_nodes)[0]
        # A = torch.tensor(A, dtype=torch.long)
        emb, pred, dir_energy, prob, edge_ratio, rewired_edge_index, updated_edge_weights = model(data.x, data.edge_index, data.batch)
        if flag == 'Test':
            visualize_rewired_graphs(data.edge_index, edge_weights=None, num_nodes=data.x.shape[0], data_name=dataname, num_layers=0, id=i, flag = 'True')
            visualize_rewired_graphs(rewired_edge_index, edge_weights=None, num_nodes=data.x.shape[0], data_name=dataname, num_layers=0, id=i, flag = 'False')
            print("Plotting done....")
        pred = pred.argmax(dim = 1)
        data.y = data.y.to(pred.device)
        correct += int((pred == data.y).sum())
        pbar.update(1)

    acc = correct / len(loader.dataset)

    return acc, dir_energy, prob, edge_ratio


parsed_args = argument_parser().parse_args()

print(parsed_args)
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
num_splits = parsed_args.num_splits
batch_size = parsed_args.batch_size
rewiring = parsed_args.rewiring
model_name = parsed_args.model 
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


if rewiring == 'gtr':
    pre_transform = T.Compose([PrecomputeGTREdges(num_edges=30)])
    transform = T.Compose([AddPrecomputedGTREdges(num_edges=30)])
    print("Applying Greedy Total Resistance rewiring")
else:
    pre_transform = None
    transform = None


train_split = 0.80
val_split = 0.10
test_split = 0.10
graph_obj = TUDatasetLoader(dataname, transform, pre_transform)
num_graphs = len(graph_obj.graphs)
train_graphs = int(num_graphs * train_split)
valid_graphs = int(num_graphs * val_split)
test_graphs = num_graphs - (train_graphs + valid_graphs)


print("Loading " + dataname)
print(f"Total graphs: {num_graphs} || Training: {train_graphs} || Validation: {valid_graphs} || Test: {test_graphs}")


for g in range(num_graphs):
    if rewiring == 'fosr':
        print("Applying FoSR rewiring")
        edge_index, edge_type, _ = fosr.edge_rewire(graph_obj.graphs[g].edge_index.numpy(), num_iterations=10)
        graph_obj.graphs[g].edge_index = torch.tensor(edge_index)
    elif rewiring == 'sdrf':
        print("Applying SDRF rewiring")
        graph_obj.graphs[g].edge_index = sdrf.sdrf(graph_obj.graphs[g], loops=10, remove_edges=False, is_undirected=True)
    elif rewiring == 'borf':
        print("Applying BORF rewiring")
        graph_obj.graphs[g].edge_index, _ = borf.borf3(graph_obj.graphs[g], 
                loops=10, 
                remove_edges=False, 
                is_undirected=True,
                batch_add=4,
                batch_remove=2,
                dataset_name=None,
                graph_index=g)
    else:
        break


test_acc_list = []
for fid in range(10, 11):
    print("----------------- Split " + str(fid) + " -------------------------")
    
    model_path = 'best_gnn_model_' + dataname + '_.pt'

    # load split, get indices, select graphs
    path = os.getcwd() + "/splits/" + dataname + "/" + dataname + "_" + str(fid) + ".npz"
    f = np.load(path)
    train_indices, val_indices, test_indices = f['arr1'], f['arr2'], f['arr3']
    train_graphs = torch.utils.data.Subset(graph_obj.graphs, train_indices)
    val_graphs = torch.utils.data.Subset(graph_obj.graphs, val_indices)
    test_graphs = torch.utils.data.Subset(graph_obj.graphs, test_indices)
    
    # train_graphs, val_graphs, test_graphs = graph_obj.load_random_splits(train_split, val_split)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle = True)

    # for i, data in enumerate(train_loader):
    #     print(data)
   
    model = GNN(graph_obj, model_name, num_layers, mlp_layers, graph_obj.num_features, hidden_dim, dropout, th, rewiring, alpha, device)
    model = model.to(device)

    opti_train = torch.optim.Adam(model.parameters(), lr=train_lr, weight_decay=train_weight_decay)

    train(train_loader, val_loader, test_loader, model, opti_train, train_iter, model_path, device)

    print("=" * 30)
    print("Model Testing....")

    avg_test_acc = 0.0
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # print("Learned threshold: ", model.prob)

    for _ in range(test_iter):
        test_acc, dir_energy, prob, edge_ratio = test(model, test_loader, flag = 'Test')
        print("Dirichlet Energy at final layer ", dir_energy)
        if rewiring  == 'simgrew':
            print("Final prob: ", prob)
            print("Final edge ratio :", edge_ratio)
            print(f"Test Accuracy: {test_acc:.4f}")
        avg_test_acc += test_acc

    avg_test_acc = avg_test_acc / test_iter 
    print(f"Average Test Accuracy: {(avg_test_acc * 100):.4f}")

    test_acc_list.append(avg_test_acc)
    # emb, pred, dir_energy,  _, _ = model(data.node_features, adj_matrix)
    # visualize(emb, data.node_labels.detach().cpu().numpy(), dataset, num_layers, fid+1)
    print("---------------------------------------------\n")
    # break
    
test_acc_list = torch.tensor(test_acc_list)
print(f'Final Test Statistics: {test_acc_list.mean():.4f} +- {test_acc_list.std():.4f}')



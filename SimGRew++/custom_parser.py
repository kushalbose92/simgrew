import argparse


def argument_parser():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help = 'enter name of dataset in smallcase', default = 'cora', type = str)
    parser.add_argument('--train_lr', help = 'training learning rate', default = 0.2, type = float)
    parser.add_argument('--seed', help = 'Random seed', default = 100, type = int)
    parser.add_argument('--num_layers', help = 'number of hidden layers', default = 2, type = int)
    parser.add_argument('--mlp_layers', help = 'number of hidden layers in mlp', default = 1, type = int)
    parser.add_argument('--hidden_dim', help = 'hidden dimension for node features', default = 16, type = int)
    parser.add_argument('--train_iter', help = 'number of training iteration', default = 100, type = int)
    parser.add_argument('--test_iter', help = 'number of test iterations', default = 1, type = int)
    parser.add_argument('--use_saved_model', help = 'use saved model in directory', default = False, type = None)
    parser.add_argument('--dropout', help = 'Dropoout in the layers', default = 0.60, type = float)
    parser.add_argument('--train_w_decay', help = 'Weight decay for the training optimizer', default = 0.0005, type = float)
    parser.add_argument('--th', help = 'supply initial edge threshold value externally', default= 1.0, type = None)
    parser.add_argument('--num_parts', help = 'number of the parts for graph samplers', default= 0, type = int)
    parser.add_argument('--num_splits', help = 'number of the splits of the dataset', default= 0, type = int)
    parser.add_argument('--batch_size', help = 'batch size', default= 0, type = int)
    parser.add_argument('--rewiring', help = 'type of the rewiring to be performed', default = None, type = None)
    parser.add_argument('--model', help = 'type of the GNN model',  default = None, type = None)
    parser.add_argument('--alpha', help = 'infusion of adjacency information', default = 0.0, type = float)
    parser.add_argument('--device', help = 'cpu or gpu device to be used', default = 'cpu', type = None)

    return parser


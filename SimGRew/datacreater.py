import numpy as np
import random
import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid

from torch_geometric.utils import remove_self_loops, homophily
from utils import * 


# datasets available --- "Cornell", "Texas", "Wisconsin"
class WebKBData():
        def __init__(self, datasetname):

                dataset = WebKB(root='data/WebKB', name= datasetname, transform = NormalizeFeatures())
                data = dataset[0]

                self.data = data
                self.name = datasetname
                self.length = len(dataset)
                self.num_features = dataset.num_features
                self.num_classes = dataset.num_classes

                self.num_nodes = data.num_nodes
                self.num_edges = data.num_edges
                self.avg_node_degree = (data.num_edges / data.num_nodes)
                self.contains_isolated_nodes = data.has_isolated_nodes()
                self.data_contains_self_loops = data.has_self_loops()
                self.is_undirected = data.is_undirected()

                self.node_features = data.x
                self.node_labels = data.y
                self.edge_index = data.edge_index

    
class ActorData():
        def __init__(self):

                dataset = Actor(root='data/Actor',  transform = NormalizeFeatures())
                data = dataset[0]
                datasetname = 'Film'
                self.data = data
                self.length = len(dataset)
                self.num_features = dataset.num_features
                self.num_classes = dataset.num_classes

                self.num_nodes = data.num_nodes
                self.num_edges = data.num_edges
                self.avg_node_degree = (data.num_edges / data.num_nodes)
                self.contains_isolated_nodes = data.has_isolated_nodes()
                self.data_contains_self_loops = data.has_self_loops()
                self.is_undirected = data.is_undirected()

                self.node_features = data.x
                self.node_labels = data.y
                self.edge_index = data.edge_index
                

class PlanetoidData():
  def __init__(self, datasetname):
    dataset = Planetoid(root = 'data/Planetoid', name = datasetname, transform=NormalizeFeatures() )
    self.data = dataset[0]
    self.node_features = self.data.x
    self.edge_index = self.data.edge_index
    self.node_labels = self.data.y
    self.num_features = dataset.num_features
    self.num_classes = dataset.num_classes

    self.num_nodes = self.data.num_nodes
    self.num_edges = self.data.num_edges
    

# datasets available --- "Crocodile", "Chemeleon", "Squirrel"
class WikipediaData():
       def __init__(self, datasetname):

                dataset = WikipediaNetwork(root='data/WikipediaNetwork', geom_gcn_preprocess=True, name= datasetname, transform = NormalizeFeatures())
                data = dataset[0]

                self.data = data
                self.name = datasetname
                self.length = len(dataset)
                self.num_features = dataset.num_features
                self.num_classes = dataset.num_classes

                self.num_nodes = data.num_nodes
                self.num_edges = data.num_edges
                self.avg_node_degree = (data.num_edges / data.num_nodes)
                self.contains_isolated_nodes = data.has_isolated_nodes()
                self.data_contains_self_loops = data.has_self_loops()
                self.is_undirected = data.is_undirected()

                self.node_features = data.x
                self.node_labels = data.y
                self.edge_index = data.edge_index





# data = WebKBData('Texas')
# data = WikipediaData('Chameleon')
# data = ActorData()
# data = PlanetoidData('Cora')

# data.edge_index = remove_self_loops(data.edge_index)[0]
# homo_ratio = homophily(data.edge_index, data.node_labels, method = 'edge')
# print("Homophily ratio ", homo_ratio)

# feat_label_ratio = feature_class_relation(data.edge_index, data.node_labels, data.node_features)
# print(f"Feature to Label ratio:  {feat_label_ratio.item(): .4f}")


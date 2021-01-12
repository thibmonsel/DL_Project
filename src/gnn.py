
#https://github.com/rusty1s/pytorch_geometric/blob/master/examples/rgcn.py
#taking into account only nodes and not edges attributes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_add_pool

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        self.hidden_channels = hidden_channels
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)
        
        self.conv1 = GraphConv(in_channels= self.hidden_channels, out_channels=64, aggr='max')
        self.conv2 = GraphConv(in_channels = 64, out_channels = 32, aggr='max')
        
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 2)
            
    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_index = data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        #print(x.shape)
        x = global_add_pool(x, data.batch)
        #print(x.shape)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        #print(x.shape)
        #print('batch size')
        #print('-----')
        return x
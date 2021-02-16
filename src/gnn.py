
#https://github.com/rusty1s/pytorch_geometric/blob/master/examples/rgcn.py
#taking into account only nodes and not edges attributes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_add_pool, global_mean_pool, global_max_pool, global_sort_pool, GlobalAttention, BatchNorm

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class Net(torch.nn.Module):
    def __init__(self, in_channels, number_hidden_layers, aggr, hidden_out_channel, out_channel, pool_layer, k=1):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.number_hidden_layers = number_hidden_layers #number of hidden GraphConv layers
        self.aggr = aggr # "add", "mean" or "max"
        self.pool_layer = pool_layer # 'add', 'max', 'mean' or 'sort'
        self.hidden_out_channel = hidden_out_channel
        self.out_channel = out_channel
        self.atom_encoder = AtomEncoder(emb_dim=self.in_channels)
        self.k = k

        
        self.graph_conv_list = nn.ModuleList()
        self.graph_conv_list.append(GraphConv(in_channels= self.in_channels, out_channels=self.hidden_out_channel, aggr=self.aggr))

        self.batchnorm = BatchNorm(in_channels=self.hidden_out_channel)

        if self.number_hidden_layers != 0 : 
            for i in range(self.number_hidden_layers):
                self.graph_conv_list.append(GraphConv(in_channels= self.hidden_out_channel, out_channels= self.hidden_out_channel, aggr=self.aggr))
                    
        self.graph_conv_list.append(GraphConv(in_channels = self.hidden_out_channel, out_channels = self.out_channel, aggr=self.aggr))
                
        if self.pool_layer == 'global_attention' :
            self.global_att = GlobalAttention(nn.Sequential(nn.Linear(self.out_channel, 10), nn.Linear(10, 1)))
         
        self.linear1 = nn.Linear(self.k*self.out_channel, 16)
        self.linear2 = nn.Linear(16, 2)
            
    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_index = data.edge_index
        
        for i, layer in enumerate(self.graph_conv_list) : 
            x = layer(x, edge_index)
            x = F.relu(x)
            if i == len(self.graph_conv_list) - 1: continue
            x = self.batchnorm(x)  
        
        
        if self.pool_layer == 'add':
            x = global_add_pool(x, data.batch)
        if self.pool_layer == 'mean':
            x = global_mean_pool(x, data.batch)
        if self.pool_layer == 'max':
            x = global_max_pool(x, data.batch)
        if self.pool_layer == 'sort':
            x = global_sort_pool(x, data.batch, self.k)
        if self.pool_layer == 'global_attention':
            x = self.global_att(x, data.batch)
            
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x

    def reset_parameters(self):
        
        #conv layer
        for graphconv in self.graph_conv_list:
            graphconv.reset_parameters()
        
        #batch norm
        self.batchnorm.reset_parameters()
        
        #global pool
        if self.pool_layer == 'global_attention':
            self.pool_layer.reset_parameters()
    
        #fully connected
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        
import torch
import torch.nn as nn
from torch.nn import Conv1d,MaxPool1d
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool, GlobalAttention, BatchNorm, GATConv

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class Net3(torch.nn.Module):
    def __init__(self, in_channels, out_channel, pool_layer, nb_heads=2, k=1):
        super(Net3, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.pool_layer = pool_layer # 'add', 'max', 'mean' or 'sort'
        self.atom_encoder = AtomEncoder(emb_dim=self.in_channels)
        self.nb_heads = nb_heads
        self.k = k

        
        self.gat1 = GATConv(in_channels= self.in_channels, out_channels=self.out_channel, heads=self.nb_heads)
        self.gat2 = GATConv(in_channels= self.nb_heads*self.out_channel, out_channels=self.out_channel,  heads=self.nb_heads)
        #self.gat3 = GATConv(in_channels= self.nb_heads*self.out_channel, out_channels=self.out_channel,  heads=self.nb_heads)
        
        if self.pool_layer == 'global_attention' :
            self.global_att = GlobalAttention(nn.Sequential(nn.Linear(self.nb_heads*self.out_channel, 10), nn.Linear(10, 1)))
        
        self.linear1 = nn.Linear(self.nb_heads*self.k*self.out_channel, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 2)
            
    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_index = data.edge_index
        x1 = F.relu(self.gat1(x, edge_index))
        x2 = F.relu(self.gat2(x1, edge_index))
        #x3 = F.relu(self.gat3(x2, edge_index))
        
        x_concat = torch.cat((x1,x2),0)
        data.batch = torch.cat((data.batch ,data.batch),0)
               
        if self.pool_layer == 'add':
            x = global_add_pool(x_concat, data.batch)
        if self.pool_layer == 'mean':
            x = global_mean_pool(x_concat, data.batch)
        if self.pool_layer == 'max':
            x = global_max_pool(x_concat, data.batch)
        if self.pool_layer == 'sort':
            x = global_sort_pool(x_concat, data.batch, self.k)
        if self.pool_layer == 'global_attention':
            x = self.global_att(x_concat, data.batch)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

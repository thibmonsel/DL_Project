import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool, GlobalAttention, GENConv

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class Net4(torch.nn.Module):
    def __init__(self, in_channels, out_channel, pool_layer, k=1):
        super(Net4, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.pool_layer = pool_layer # 'add', 'max', 'mean' or 'sort'
        self.atom_encoder = AtomEncoder(emb_dim=self.in_channels)
        self.k = k

        self.genconv1 = GENConv(in_channels= self.in_channels, out_channels=self.out_channel)
        self.genconv2 = GENConv(in_channels= self.out_channel, out_channels=self.out_channel)
        self.genconv3 = GENConv(in_channels= self.out_channel, out_channels=self.out_channel)
        self.genconv4 = GENConv(in_channels= self.out_channel, out_channels=self.out_channel)

        if self.pool_layer == 'global_attention' :
            self.global_att = GlobalAttention(nn.Sequential(nn.Linear(self.out_channel, 10), nn.Linear(10, 1)))
        
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 2)
            
    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_index = data.edge_index
        
        x = F.relu(self.genconv1(x, edge_index))
        x = F.relu(self.genconv2(x, edge_index))
        x = F.relu(self.genconv3(x, edge_index))
        x = F.relu(self.genconv4(x, edge_index))
        
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
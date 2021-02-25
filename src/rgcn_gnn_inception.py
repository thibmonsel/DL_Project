
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, FastRGCNConv, global_add_pool, global_mean_pool, global_max_pool, global_sort_pool, GlobalAttention, BatchNorm

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

NUM_RELATIONS = 60

class Net(torch.nn.Module):
    def __init__(self, in_channels, number_hidden_layers, aggr, hidden_out_channel, out_channel, pool_layer, k=1, device=None):
        super(Net, self).__init__()
        self.pool_layer = pool_layer # 'add', 'max', 'mean' or 'sort'
        self.device = device
        self.k = k
        self.atom_encoder = AtomEncoder(emb_dim=in_channels)
        self.batchnorm = BatchNorm(in_channels=2*hidden_out_channel)

        self.rgcn_list = torch.nn.ModuleList()
        self.graphconv_list = torch.nn.ModuleList()
        self.rgcn_list.append(FastRGCNConv(in_channels=in_channels, out_channels=hidden_out_channel, num_relations=NUM_RELATIONS))
        self.graphconv_list.append(GraphConv(in_channels=in_channels, out_channels=hidden_out_channel))

        if number_hidden_layers != 0 : 
            for i in range(number_hidden_layers):
                self.rgcn_list.append(FastRGCNConv(in_channels=2*hidden_out_channel, out_channels=hidden_out_channel, num_relations=NUM_RELATIONS))
                self.graphconv_list.append(GraphConv(in_channels=2*hidden_out_channel, out_channels=hidden_out_channel))

             
        self.rgcn_list.append(FastRGCNConv(in_channels=2*hidden_out_channel, out_channels=out_channel, num_relations=NUM_RELATIONS))
        self.graphconv_list.append(GraphConv(in_channels=2*hidden_out_channel, out_channels=out_channel))


        self.linear1 = nn.Linear(2*k*out_channel, 16)
        self.linear2 = nn.Linear(16, 1)

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_attr = torch.LongTensor([ edge_type[0] + edge_type[1]*5 + edge_type[2]*30 for edge_type in edge_attr]).to(self.device)
        for i in range(len(self.rgcn_list)): 
            x_rgcn = self.rgcn_list[i](x, edge_index,edge_attr)
            x_gconv = self.graphconv_list[i](x, edge_index)
            x= torch.cat((x_rgcn,x_gconv),1)
            x = F.relu(x)
            if i == len(self.rgcn_list) - 1: continue
            x = self.batchnorm(x)  
        
        # x = self.graph_conv(x,edge_index)
        # x = F.relu(x)

        if self.pool_layer == 'add':
            x = global_add_pool(x, data.batch)
        if self.pool_layer == 'mean':
            x = global_mean_pool(x, data.batch)
        if self.pool_layer == 'max':
            x = global_max_pool(x, data.batch)
        if self.pool_layer == 'sort':
            x = global_sort_pool(x, data.batch, self.k)
   
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x

    def reset_parameters(self):
        
        #conv layer
        for rgcn in self.rgcn_list:
            rgcn.reset_parameters()
        
        #batch norm
        self.batchnorm.reset_parameters()

        # self.graph_conv.reset_parameters()
    
        #fully connected
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()



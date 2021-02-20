import torch

#Network parameters

#example of params for gnn model
gnn_params = {
    'in_channel' : 100,
    'number_hidden_layers' : 2,
    'aggregator' : 'add', # ['add', 'max', 'mean']
    'hidden_out_channel' : 64,
    'out_channel' : 32,
    'pool_layer' : 'add', # ['add', 'max', 'mean', 'sort']
    'k' : 1, # changed when using sort
}

#example of params for rcgn model
rgcn_params = {
    'in_channel' : 100,
    'number_hidden_layers' : 2,
    'aggregator' : 'add', # ['add', 'max', 'mean']
    'hidden_out_channel' : 64,
    'out_channel' : 32,
    'pool_layer' : 'add', # ['add', 'max', 'mean', 'sort']
    'k' : 1, # changed when using sort
    'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}
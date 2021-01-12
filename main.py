import os, sys 
sys.path.append("./src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GraphConv

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ignite.contrib.metrics.roc_auc import roc_auc_compute_fn
from gnn import Net

dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root='../')
 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=64, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net(100)
model.to(device) # puts model on GPU / CPU

# optimization hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05) # try lr=0.01, momentum=0.9
loss_fn = nn.CrossEntropyLoss()

# main loop (train+test)
for epoch in range(2):
    # training
    model.train() # mode "train" agit sur "dropout" ou "batchnorm"
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        data.y = data.y.view(-1)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        if batch_idx %100 ==0:
            print('epoch {} batch {} [{}/{}] training loss: {}'.format(epoch,batch_idx,batch_idx*len(data),
                    len(train_loader.dataset),loss.item()))
    
    # testing
    model.eval()
    correct, concat_prediction, concat_target = 0, torch.empty(0), torch.empty(0)
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            optimizer.zero_grad()
            data = data.to(device)
            data.y = data.y.flatten()
            out = model(data)
            loss = loss_fn(out, data.y)
            prediction = out.argmax(dim=1, keepdim=True) 
            concat_prediction = torch.cat((concat_prediction, prediction), 0)
            concat_target = torch.cat((concat_target, data.y), 0)
            correct += prediction.eq(data.y.view_as(prediction)).sum().item()
    taux_classif = 100. * correct / len(test_loader.dataset)
    print('Accuracy: {} --- ROC_AUC {} \n'.format(taux_classif, roc_auc_compute_fn(concat_prediction, concat_target)))
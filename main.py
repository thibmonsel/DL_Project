import os, sys 
sys.path.append("./src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GraphConv
from torch.optim.lr_scheduler import MultiplicativeLR

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ignite.contrib.metrics.roc_auc import roc_auc_compute_fn
import matplotlib.pyplot as plt
from gnn import Net, Net2

dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root='../')
 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=64, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=64, shuffle=False)

# With have 3.5% of positive label in our dataset

IN_CHANNELS = 100
NUMBER_HIDDEN_LAYERS = 1
AGGR = ['add', 'max', 'mean']
HIDDEN_OUT_CHANNEL = 64
OUT_CHANNEL = 32
POOL_LAYERS = ['add', 'max', 'mean', 'sort', 'global_attention']
k = 3
EPOCHS = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net(IN_CHANNELS, NUMBER_HIDDEN_LAYERS, AGGR[0], HIDDEN_OUT_CHANNEL, OUT_CHANNEL, POOL_LAYERS[0])
model.to(device) # puts model on GPU / CPU

print("Model's architecture is : {} \n".format(model))
# optimization hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05) # try lr=0.01, momentum=0.9
scheduler = MultiplicativeLR(optimizer, lr_lambda= lambda epoch : 0.95) # multiplies lr by 0.95 at each epoch
loss_fn = nn.CrossEntropyLoss()

print("## TRAINING ##")
# main loop (train+test)
loss_train, loss_val = [], []
for epoch in range(EPOCHS):
    # training
    model.train() # mode "train" agit sur "dropout" ou "batchnorm"
    running_train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        data.y = data.y.view(-1)
        loss = loss_fn(out, data.y)
        running_train_loss += loss.item() * data.x.size(0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx %100 ==0:
            print('epoch {} batch {} [{}/{}] training loss: {}'.format(epoch,batch_idx,batch_idx*len(data),
                    len(train_loader.dataset),loss.item()))
    loss_train.append(running_train_loss / len(train_loader.dataset))
    
    print("## VALIDATING ##")
    model.eval()
    correct, concat_prediction, concat_target, running_val_loss = 0, torch.empty(0).to(device), torch.empty(0).to(device), 0
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            data = data.to(device)
            data.y = data.y.view(-1)
            out = model(data)
            loss = loss_fn(out, data.y)
            running_val_loss += loss.item() * data.x.size(0)
            prediction = out.argmax(dim=1, keepdim=True) 
            concat_prediction = torch.cat((concat_prediction, prediction), 0)
            concat_target = torch.cat((concat_target, data.y), 0)
            correct += prediction.eq(data.y.view_as(prediction)).sum().item()
    roc_auc = roc_auc_compute_fn(concat_prediction.to("cpu"), concat_target.to("cpu"))
    loss_val.append(running_val_loss / len(valid_loader.dataset))
    print('ROC_AUC score : {} \n'.format(roc_auc))
    
plt.title("Training and validation loss curves")
plt.plot(loss_train, 'go-',label="train")
plt.plot(loss_val, 'rs', label='val')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("plots/GraphConv_epoch{}_inChannel{}_numHiddenLayers{}_aggr{}_hiddenOutChannel{}_globalPool{}.png".format(EPOCHS, IN_CHANNELS, NUMBER_HIDDEN_LAYERS, AGGR[0], HIDDEN_OUT_CHANNEL, OUT_CHANNEL, POOL_LAYERS[0]))
plt.show()

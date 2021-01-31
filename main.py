import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GraphConv
import matplotlib.pyplot as plt

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from ignite.contrib.metrics.roc_auc import roc_auc_compute_fn
from src.vanillagnn import VanNet
from src.gnn import Net

# train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), stratify = labels, shuffle=True, test_size=0.3)
# train_idx.sort(), test_idx.sort()
# train_idx, test_idx = torch.from_numpy(train_idx), torch.from_numpy(test_idx)

# #stratified test train split but not stratified batches
# train_loader = DataLoader(dataset[train_idx],  batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset[test_idx] ,batch_size=64, shuffle=False)

# With have 3.5% of positive label in our dataset

def train(model, device, train_loader, loss_fn, optimizer):
    running_train_loss = 0
    model.train()
    for step, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        data.y = data.y.view(-1)
        loss = loss_fn(out, data.y)
        running_train_loss +=loss.item()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        if step % 100 ==0:
            print('batch {} [{}/{}] training loss: {}'.format(step,step*64,
                    len(train_loader.dataset),loss.item()))
    return running_train_loss / len(train_loader.dataset)

def eval(model,device, loader, evaluator):
    running_eval_loss = 0
    model.eval()
    concat_prediction, concat_target = torch.empty(0), torch.empty(0)
    with torch.no_grad():
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            data.y = data.y.flatten()
            out = model(data)
            loss = loss_fn(out, data.y)
            running_eval_loss +=loss.item()
            prediction = out.argmax(dim=1, keepdim=True) 
            concat_prediction = torch.cat((concat_prediction, prediction), 0)
            concat_target = torch.cat((concat_target, data.y), 0)
        input_dict = {"y_true": concat_target.unsqueeze(1).numpy(), "y_pred": concat_prediction.numpy()}
    return evaluator.eval(input_dict)['rocauc'], running_eval_loss / len(loader.dataset)

        
if __name__ == '__main__':
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root='../')
    evaluator = Evaluator("ogbg-molhiv")    

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    IN_CHANNELS = 100
    NUMBER_HIDDEN_LAYERS = 2
    AGGR = ['add', 'max', 'mean']
    HIDDEN_OUT_CHANNEL = 64
    OUT_CHANNEL = 32
    POOL_LAYERS = ['add', 'max', 'mean', 'sort', 'global_attention']
    k = 3
    EPOCHS = 100

    #model = VanNet(100)
    model = Net(IN_CHANNELS, NUMBER_HIDDEN_LAYERS, AGGR[1], HIDDEN_OUT_CHANNEL, OUT_CHANNEL, POOL_LAYERS[0])
    model.to(device)

    # optimization hyperparameters
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.05) 
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1, 3]))
    #scheduler = MultiplicativeLR(optimizer, lr_lambda= lambda epoch : 0.95)

    results = {'highest_valid': 0,
            'final_train': 0,
            'final_test': 0,
            'highest_train': 0}
    
    test_l, train_l = [], []
    for epoch in range(10):
        print("Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, loss_fn, optimizer) 
        
        print("Evaluating...")
        train_roc, train_loss= eval(model,device, train_loader, evaluator)
        valid_roc, valid_loss =  eval(model,device, valid_loader, evaluator)
        test_roc, test_loss= eval(model, device, test_loader, evaluator)
        print("Train  ROC_AUC score : {}".format(train_roc))
        print("Validation ROC_AUC score : {}".format(valid_roc))
        
        test_l.append(test_loss), train_l.append(train_loss)
        
        if train_roc > results['highest_train']:
            results['highest_train'] = train_roc

        if valid_roc > results['highest_valid']:
            results['highest_valid'] = valid_roc
            results['final_train'] = train_roc
            results['final_test'] = test_roc

print("Post training Results {}".format(results))
plt.title("Training and validation loss curves")
plt.plot(train_l, 'go-',label="train")
plt.plot(test_l, 'rs-', label='val')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plots/GraphConv_epoch{}_inChannel{}_numHiddenLayers{}_aggr{}_hiddenOutChannel{}_globalPool{}.png".format(EPOCHS, IN_CHANNELS, NUMBER_HIDDEN_LAYERS, AGGR[0], HIDDEN_OUT_CHANNEL, OUT_CHANNEL, POOL_LAYERS[2]))
plt.show()

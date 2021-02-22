import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GraphConv
import matplotlib.pyplot as plt

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader

from data_loader import TrainValTestLoader

from params import gnn_params, rgcn_params
from src.gnn import GCN_Net
from src.rgcn import Net

# With have 3.5% of positive label in our dataset

def train(model, device, train_loader, loss_fn, optimizer):
    model.train()
    
    running_train_loss = 0
    for step, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = loss_fn(out, data.y.to(out.dtype))
        running_train_loss += loss.item() * len(data.y)
        loss.backward()
        optimizer.step()
    return running_train_loss / len(train_loader.dataset)

def eval(model,device, loader, evaluator):
    model.eval()
        
    running_eval_loss = 0
    concat_prediction, concat_target = torch.empty(0,1), torch.empty(0,1)
    with torch.no_grad():
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y.to(out.dtype))
            running_eval_loss += loss.item() * len(data.y)
            concat_prediction = torch.cat((concat_prediction, out.cpu()), 0)
            concat_target = torch.cat((concat_target, data.y.cpu()), 0)
        
        input_dict = {"y_true": concat_target.numpy(), "y_pred": concat_prediction.numpy()}
    return evaluator.eval(input_dict)['rocauc'], running_eval_loss / len(loader.dataset)

        
if __name__ == '__main__':
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root='../')
    evaluator = Evaluator("ogbg-molhiv")    

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 25
    
    parameters = gnn_params # or gnn_params
    model = GCN_Net(*parameters.values())
    #model = Net(*parameters.values())
    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    test_perfs = []
    for run in range(1,11) :
        print(f'Run {run}:')

        model.reset_parameters()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.05) 
        
        results = {'highest_valid': 0,
                'final_train': 0,
                'final_test': 0,
                'highest_train': 0}
        
        for epoch in range(EPOCHS):
            loss = train(model, device, train_loader, loss_fn, optimizer) 
            train_roc, _ = eval(model,device, train_loader, evaluator)
            valid_roc, _ = eval(model,device, valid_loader, evaluator)
            test_roc, _ = eval(model, device, test_loader, evaluator)
                        
            if train_roc > results['highest_train']:
                results['highest_train'] = train_roc

            if valid_roc > results['highest_valid']:
                results['highest_valid'] = valid_roc
                results['final_train'] = train_roc
                results['final_test'] = test_roc
            
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_roc:.4f}, Val: {valid_roc:.4f}, '
              f'Test: {test_roc:.4f}') 
             
        print("Post training Results {}".format(results))

        test_perfs.append(results['final_test'])

test_perf = torch.tensor(test_perfs)
print('===========================')
print("model parameters : {}".format(parameters))
print(f'Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}')



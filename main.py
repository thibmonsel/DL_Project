import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GraphConv
from torch.optim.lr_scheduler import MultiplicativeLR

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from src.gnn import Net
from src.chebConv import Net2
from src.GATConv import Net3
from src.GENConv import Net4


# train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), stratify = labels, shuffle=True, test_size=0.3)
# train_idx.sort(), test_idx.sort()
# train_idx, test_idx = torch.from_numpy(train_idx), torch.from_numpy(test_idx)

# #stratified test train split but not stratified batches
# train_loader = DataLoader(dataset[train_idx],  batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset[test_idx] ,batch_size=64, shuffle=False)

# With have 3.5% of positive label in our dataset

def train(model, device, loader, loss_fn, optimizer, scheduler):
    model.train()
    running_train_loss = 0
    for step, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        data.y = data.y.view(-1)
        loss = loss_fn(out, data.y)
        running_train_loss += loss.item() * data.x.size(0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 ==0:
            print('batch {} [{}/{}] training loss: {}'.format(step,step*64,
                    len(loader.dataset),loss.item()))
    return running_train_loss / len(loader.dataset)

@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    concat_prediction, concat_target = torch.empty(0).to(device), torch.empty(0).to(device)
    for step, data in enumerate(loader):
        data = data.to(device)
        data.y = data.y.view(-1)
        out = model(data)
        prediction = out.argmax(dim=1, keepdim=True) 
        print("pred", prediction)
        concat_prediction = torch.cat((concat_prediction, prediction), 0)
        concat_target = torch.cat((concat_target, data.y), 0)
        
    input_dict = {"y_true": concat_target.unsqueeze(1).numpy(), "y_pred": concat_prediction.numpy()}
    return evaluator.eval(input_dict)['rocauc']
    
if __name__ == '__main__':
    
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root='../')
    evaluator = Evaluator("ogbg-molhiv")    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=64, shuffle=False)
    
    IN_CHANNELS = 100
    NUMBER_HIDDEN_LAYERS = 1
    AGGR = ['add', 'max', 'mean']
    HIDDEN_OUT_CHANNEL = 64
    OUT_CHANNEL = 32
    POOL_LAYERS = ['add', 'max', 'mean', 'sort', 'global_attention']
    k = 3
    EPOCHS = 100

    model = Net4(IN_CHANNELS, OUT_CHANNEL, POOL_LAYERS[1])
    #model = Net3(IN_CHANNELS, 100, 'max')
    #model = Net2(IN_CHANNELS, HIDDEN_OUT_CHANNEL, OUT_CHANNEL, POOL_LAYERS[0])
    #model = Net(IN_CHANNELS, NUMBER_HIDDEN_LAYERS, AGGR[0], HIDDEN_OUT_CHANNEL, OUT_CHANNEL, POOL_LAYERS[0])
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1) 
    scheduler = MultiplicativeLR(optimizer, lr_lambda= lambda epoch : 0.95)
    loss_fn = nn.CrossEntropyLoss()

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}
    
    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, loss_fn, optimizer, scheduler) 

        print("Evaluating...")
        #train_roc = eval(model, device, train_loader, evaluator)
        valid_roc = eval(model, device, valid_loader, evaluator)
        #test_roc = eval(model, device, test_loader, evaluator)
        print("Validation ROC_AUC score : {}".format(valid_roc))

        # if train_roc > results['highest_train']:
        #     results['highest_train'] = train_roc

        # if valid_roc > results['highest_valid']:
        #     results['highest_valid'] = valid_roc
        #     results['final_train'] = train_roc
        #     results['final_test'] = test_roc

    print("Post training Results {}".format(results))
    # plt.title("Training and validation loss curves")
    # plt.plot(loss_train, 'go-',label="train")
    # plt.plot(loss_val, 'rs', label='val')
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig("plots/GraphConv_epoch{}_inChannel{}_numHiddenLayers{}_aggr{}_hiddenOutChannel{}_globalPool{}.png".format(EPOCHS, IN_CHANNELS, NUMBER_HIDDEN_LAYERS, AGGR[0], HIDDEN_OUT_CHANNEL, OUT_CHANNEL, POOL_LAYERS[2]))
    # plt.show()
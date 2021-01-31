import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


class TrainValTestLoader : 
    
    def __init__(self, dataset, test_train_split=0.8, val_test_split=0.5, shuffle=False):
        self.dataset = dataset
        self.test_train_split = test_train_split
        self.val_test_split = val_test_split
        self.dataset_size = len(self.dataset)
        self.labels = self.dataset.data.y.numpy()
        self.positive_indices =  np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]
        
        if shuffle:
            np.random.shuffle(self.positive_indices)
            np.random.shuffle(self.negative_indices)
    
    def get_train_val_test_splitters(self):  
        #positive datapoints      
        pos_indices = np.arange(len(self.positive_indices))
        pos_test_split = int(np.floor(self.test_train_split * len(self.positive_indices)))
        pos_train_indices, pos_test_indices = pos_indices[:pos_test_split], pos_indices[pos_test_split:] 
        
        pos_valid_split = int(np.floor(self.val_test_split * len(pos_test_indices)))
        pos_test_indices, pos_val_indices = pos_test_indices[:pos_valid_split], pos_test_indices[pos_valid_split:] 
        pos_dic = dict(zip(["pos_train", "pos_valid", "pos_test"],[np.take(self.positive_indices, pos_train_indices), 
                                                                   np.take(self.positive_indices, pos_test_indices),
                                                                   np.take(self.positive_indices, pos_val_indices)]))
        
        #negative datapoints
        neg_indices = np.arange(len(self.negative_indices))
        neg_test_split = int(np.floor(self.test_train_split * len(self.negative_indices)))
        neg_train_indices, neg_test_indices = neg_indices[:neg_test_split], neg_indices[neg_test_split:] 
        neg_valid_split = int(np.floor(self.val_test_split * len(neg_test_indices)))
        neg_test_indices, neg_val_indices = neg_test_indices[:neg_valid_split], neg_test_indices[neg_valid_split:] 
        neg_dic = dict(zip(["neg_train", "neg_valid", "neg_test"],[np.take(self.negative_indices, neg_train_indices), 
                                                                   np.take(self.negative_indices, neg_test_indices), 
                                                                   np.take(self.negative_indices, neg_val_indices)]))
        
        assert(len(pos_train_indices) +  len(pos_test_indices) + len(pos_val_indices) + len(neg_train_indices) +  len(neg_test_indices) + len(neg_val_indices) == self.dataset_size)
        return neg_dic, pos_dic
    
    def merge_positive_negative_indices(self):
        neg_dic, pos_dic = self.get_train_val_test_splitters()
        return dict(zip(["train", "valid", "test"], [torch.from_numpy(np.concatenate((pos_dic["pos_train"], neg_dic["neg_train"]), axis=0)),
                                                     torch.from_numpy(np.concatenate((pos_dic["pos_valid"], neg_dic["neg_valid"]), axis=0)), 
                                                     torch.from_numpy(np.concatenate((pos_dic["pos_test"], neg_dic["neg_test"]), axis=0))]))
